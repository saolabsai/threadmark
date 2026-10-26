"""Deterministic, synthetic multi-entity workflows; never calls or trains a model.

Public events/catalog/questions are separate from construction labels. Test
generation requires a caller-supplied locked protocol hash. Development uses a
different sentence family, but shares scenario structure: this is a synthetic
held-out-instance test, not a test of unseen scenario families or real emails.

Adapted from the published research generator. This portable copy has its own
source hash and version; new output is not the frozen published benchmark.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re

VERSION = "threadmark-synthetic-workflows-v1"
DEFAULT_SEED = 20261027
DOMAINS = ("meeting", "support", "project")
SIZE_BANDS = (2048, 8192, 16384)
TOLERANCE = 0.10
EVIDENCE_CONTRACT = (
    "For each current fact, cite its latest direct setting or confirming message. "
    "An explicit unknown or cleared fact has value null and cites that message. "
    "A field never established has value null and an empty evidence list. "
    "Proposals and drafts do not overwrite approved facts. A message changes only "
    "the fields it explicitly states; keep other fields unchanged."
)

FIELD_DEFINITIONS = {
    "meeting": {
        "status": "Current booking state: proposed, confirmed, or cancelled.",
        "start_at": "Agreed start as YYYY-MM-DDTHH:MM:SS+00:00, or null if none. All times in this thread are UTC.",
        "timezone": "IANA timezone, UTC in these workflows, or null if unstated.",
        "owner": "Exact full name of the current coordinator, or null.",
        "location": "Exact agreed venue label, or null if not selected or cleared.",
        "pending_start_at": "Unaccepted proposed start as YYYY-MM-DDTHH:MM:SS+00:00, or null.",
        "unresolved_decision": "schedule_approval, venue_selection, or null when no decision is pending.",
    },
    "support": {
        "status": "Current ticket state: open, resolved, or reopened.",
        "owner": "Exact full name of the current ticket owner, or null.",
        "priority": "Current approved priority: low, normal, high, or urgent.",
        "response_due_at": "Agreed next response deadline as YYYY-MM-DDTHH:MM:SS+00:00, or null if none.",
        "approved_refund": "Approved refund as USD N.NN, or null if no amount is approved.",
        "pending_refund": "Unapproved refund request as USD N.NN, or null if none.",
        "resolution": "Exact quoted resolution label, or null when no current resolution exists.",
        "unresolved_decision": "refund_approval, engineering_review, or null when no decision is pending.",
    },
    "project": {
        "status": "Current workstream state: planned, active, blocked, cancelled, or completed.",
        "owner": "Exact full name of the accountable workstream owner, or null.",
        "due_date": "Approved delivery date as YYYY-MM-DD, or null if none.",
        "budget_cap": "Approved spending ceiling as USD N.NN, or null.",
        "approved_scope": "Exact quoted approved scope package label, or null.",
        "pending_scope": "Exact quoted scope package awaiting approval, or null.",
        "dependency": "Exact other workstream label on which this workstream currently depends, or null.",
        "unresolved_decision": "scope_approval, vendor_choice, or null when no decision is pending.",
    },
}
NAMES = (
    "Mira Shah",
    "Leon Park",
    "Amara Cole",
    "Theo Reed",
    "Nora Patel",
    "Ivo Chen",
    "Sofia Diaz",
    "Arun Moss",
)
LABEL_WORDS = (
    "Alder",
    "Beacon",
    "Cedar",
    "Delta",
    "Ember",
    "Falcon",
    "Grove",
    "Harbor",
    "Iris",
    "Juniper",
    "Kestrel",
    "Linden",
)
VENUES = (
    "North conference room",
    "Canal studio",
    "West video room",
    "Library suite",
    "Garden conference room",
)
SCOPES = (
    "accessibility review",
    "billing dashboard",
    "audit export",
    "partner onboarding",
    "search filters",
    "notification center",
)
CONTEXT = {
    "meeting": (
        "The presenter is combining the customer questions with the engineering notes so that we can discuss one coherent agenda. The materials cover handoffs, open assumptions, and the evidence needed for a decision. Please keep the pre-read focused on those issues; general product news can go in the written appendix.",
        "Several attendees need to review the same diagrams before the discussion. The diagrams show where responsibility passes between teams and which checks happen before a release. The meeting should leave enough space to compare alternatives, because the first suggestion in the document is still only a working example.",
        "The operations team is collecting questions from people who cannot attend. We will preserve the questions in the agenda and record answers afterward. A short explanation of each tradeoff will be more useful than a long slide deck, particularly for colleagues joining the discussion without the earlier project background.",
        "The pre-read now explains the assumptions behind the capacity figures and separates measured observations from estimates. Reviewers should note which claims need a follow-up experiment. The discussion is intended to make those uncertainties visible, with enough background for participants from both the service team and the product team.",
    ),
    "support": (
        "The customer supplied a timeline of the failed attempts and a short screen recording. The symptoms vary between the browser and the exported report, so we are comparing both paths. The troubleshooting notes should describe what was actually observed and avoid treating a plausible explanation as a verified cause.",
        "The account uses the service during several different shifts. Their team needs a concise explanation that can be passed between operators without repeating the whole investigation. We are keeping the reproduction steps beside the observed behavior so the next person can see exactly which part remains uncertain.",
        "The diagnostic bundle contains request identifiers, a description of the affected workflow, and a redacted sample of the output. It does not contain credentials. The customer cares most about whether the workaround preserves their usual checks, so the response should explain both the benefit and the remaining limitation.",
        "The support notes distinguish the immediate symptom from the wider account history. A prior case involved similar wording but a different failure path. We are comparing the observations carefully before reusing any explanation, and keeping a record of which tests were performed rather than assuming the previous fix applies.",
    ),
    "project": (
        "The implementation notes separate the customer-visible behavior from the internal migration work. Reviewers need to understand which acceptance checks protect existing users and which checks cover the new path. The design discussion should preserve that distinction so a convenient implementation detail does not silently become a product requirement.",
        "The team is comparing a gradual rollout with a single release. The decision depends on how quickly we can detect regressions and how easily an operator can reverse a change. The planning document captures those tradeoffs, including the manual work that is easy to miss when looking only at the code estimate.",
        "The review packet includes interface examples and a small list of operational failure modes. Each example is intended to make the proposed behavior concrete rather than prescribe the implementation. We will use the same examples during acceptance so the discussion does not drift toward features that were never requested.",
        "The integration notes explain which assumptions are controlled by our team and which depend on outside systems. The next review should challenge those assumptions with concrete examples. We are keeping investigation results separate from estimates, since an optimistic estimate should not be mistaken for evidence that a dependency is ready.",
    ),
}


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def render_events(events):
    return "\n\n".join(
        f"[{e['event_id']}] {e['timestamp']} From: {e['sender']}\n{e['text']}"
        for e in events
    )


def public_workflow(workflow):
    """Explicit inference allowlist, deliberately excluding labels and metrics."""
    return {
        "id": workflow["id"],
        "domain": workflow["domain"],
        "entity_catalog": deepcopy(workflow["entity_catalog"]),
        "events": deepcopy(workflow["events"]),
        "queries": [
            {k: deepcopy(q[k]) for k in ("query_id", "text", "requested_facts")}
            for q in workflow["queries"]
        ],
    }


class Thread:
    def __init__(self, identifier, domain, split, rng, token_counter):
        self.id, self.domain, self.split, self.rng, self.count = (
            identifier,
            domain,
            split,
            rng,
            token_counter,
        )
        self.people = rng.sample(NAMES, len(NAMES))
        words = rng.sample(LABEL_WORDS, 2)
        suffix = {"meeting": "review", "support": "ticket", "project": "workstream"}[
            domain
        ]
        self.labels = [f"{word} {suffix} {rng.randrange(100, 999)}" for word in words]
        self.catalog = [
            {
                "entity_id": f"item-{i+1}",
                "label": label,
                "kind": domain,
                "fields": list(FIELD_DEFINITIONS[domain]),
                "field_definitions": dict(FIELD_DEFINITIONS[domain]),
                "evidence_contract": EVIDENCE_CONTRACT,
            }
            for i, label in enumerate(self.labels)
        ]
        self.events, self.updates, self.event_metrics = [], [], []
        self.state = {
            (e["entity_id"], field): {
                "entity_id": e["entity_id"],
                "field": field,
                "value": None,
                "evidence_event_ids": [],
            }
            for e in self.catalog
            for field in e["fields"]
        }
        self.day_offset = rng.randrange(0, 20)

    def date(self, step, hour=10):
        moment = datetime(2027, 3, 1, hour, tzinfo=timezone.utc) + timedelta(
            days=self.day_offset + step
        )
        return moment.isoformat(timespec="seconds")

    def human(self, date):
        return datetime.fromisoformat(date).strftime("%d %B %Y at %H:%M UTC")

    def amount(self, step):
        return f"USD {100 + self.day_offset * 25 + step * 50}.00"

    def wording(self, test, dev):
        return test if self.split == "test" else dev

    def emit(self, core, updates=(), *, context=True, category="decision"):
        index = len(self.events)
        event_id = f"e{index+1:04d}"
        body = core
        ancillary = ""
        if context:
            ancillary = CONTEXT[self.domain][index % len(CONTEXT[self.domain])]
            # The contextual subject ties the explanation to a real event, but
            # this finite ancillary family is disclosed and not called unique prose.
            body += "\n\n" + ancillary
        self.events.append(
            {
                "event_id": event_id,
                "timestamp": (
                    datetime(2027, 1, 4, 8, tzinfo=timezone.utc)
                    + timedelta(minutes=17 * index)
                ).isoformat(timespec="seconds"),
                "sender": self.people[index % len(self.people)],
                "text": body,
            }
        )
        changed, facts = 0, []
        for entity_index, field, value in updates:
            key = (f"item-{entity_index+1}", field)
            changed += self.state[key]["value"] != value
            fact = {
                "entity_id": key[0],
                "field": field,
                "value": value,
                "evidence_event_ids": [event_id],
            }
            self.state[key] = fact
            facts.append(deepcopy(fact))
        self.updates.append({"event_id": event_id, "facts": facts})
        self.event_metrics.append(
            {
                "event_id": event_id,
                "category": category,
                "set_fact_count": len(facts),
                "changed_fact_count": changed,
                "decision_text_tokens": self.count(core) if facts else 0,
                "nonoperative_text_tokens": self.count(core) if not facts else 0,
                "ancillary_text_tokens": self.count(ancillary) if ancillary else 0,
            }
        )

    def initialize(self):
        for i, label in enumerate(self.labels):
            owner, start, amount = self.people[i], self.date(i), self.amount(i)
            lead = self.wording(
                f"Here is the agreed starting position for {label}.",
                f"Opening record for {label}: these arrangements have been accepted.",
            )
            if self.domain == "meeting":
                text = f"{lead} The booking is confirmed for {self.human(start)}, in {VENUES[i]}. {owner} is the coordinator. The timezone is UTC. There is no alternative start pending and no unresolved decision."
                values = dict(
                    status="confirmed",
                    start_at=start,
                    timezone="UTC",
                    owner=owner,
                    location=VENUES[i],
                    pending_start_at=None,
                    unresolved_decision=None,
                )
            elif self.domain == "support":
                text = f"{lead} The ticket is open, assigned to {owner}, with normal priority. Our next response is due {self.human(start)}. A refund of {amount} is approved. There is no pending refund request, no current resolution, and no unresolved decision."
                values = dict(
                    status="open",
                    owner=owner,
                    priority="normal",
                    response_due_at=start,
                    approved_refund=amount,
                    pending_refund=None,
                    resolution=None,
                    unresolved_decision=None,
                )
            else:
                text = f"{lead} Work is active under {owner}, with delivery approved for {start[:10]} and a spending ceiling of {amount}. The approved scope package is '{SCOPES[i]}'. There is no pending scope request, no dependency, and no unresolved decision."
                values = dict(
                    status="active",
                    owner=owner,
                    due_date=start[:10],
                    budget_cap=amount,
                    approved_scope=SCOPES[i],
                    pending_scope=None,
                    dependency=None,
                    unresolved_decision=None,
                )
            self.emit(text, [(i, field, value) for field, value in values.items()])

    def middle(self, step):
        i, phase = step % 2, (step // 2) % 6
        label = self.labels[i]
        owner = self.people[(step // 2 + 2 + i) % len(self.people)]
        start, amount = self.date(step + 3, 9 + step % 6), self.amount(step + 4)
        lead = self.wording(
            f"Update for {label}:",
            f"Please amend the live record of {label} as follows.",
        )
        updates = []
        if phase == 0:
            # Owners are deliberately retained from early/middle sources. Other
            # fields continue to change so long threads are not neutral padding.
            if self.domain == "meeting":
                venue = VENUES[(step // 2 + 2) % len(VENUES)]
                core = f"{lead} The participants have approved {venue} as the venue, replacing the earlier room."
                updates = [(i, "location", venue)]
            elif self.domain == "support":
                priority = ("high", "urgent", "low")[(step // 12 + i) % 3]
                core = f"{lead} The supervisor approved {priority} priority, replacing the previous priority."
                updates = [(i, "priority", priority)]
            else:
                core = f"{lead} The sponsor approved a revised spending ceiling of {amount}, replacing the previous ceiling."
                updates = [(i, "budget_cap", amount)]
        elif phase == 1:
            if self.domain == "meeting":
                core = f"{lead} The participants have approved moving the confirmed start to {self.human(start)}. The prior time is obsolete; the booking remains confirmed. Any previous start-time proposal is now closed, with no unresolved decision."
                updates = [
                    (i, "start_at", start),
                    (i, "status", "confirmed"),
                    (i, "pending_start_at", None),
                    (i, "unresolved_decision", None),
                ]
            elif self.domain == "support":
                core = f"{lead} The customer accepted a revised next-response deadline of {self.human(start)}. This supersedes the previous deadline."
                updates = [(i, "response_due_at", start)]
            else:
                core = f"{lead} The sponsor approved {start[:10]} as the delivery date, replacing the earlier target. The spending ceiling is also revised and approved at {amount}."
                updates = [(i, "due_date", start[:10]), (i, "budget_cap", amount)]
        elif phase == 2:
            if self.domain == "meeting":
                core = f"{lead} A move to {self.human(start)} has been requested but not accepted. Keep the agreed booking intact. Record this as the pending alternative; schedule approval is the unresolved decision."
                updates = [
                    (i, "pending_start_at", start),
                    (i, "unresolved_decision", "schedule_approval"),
                ]
            elif self.domain == "support":
                core = f"{lead} The customer requests a refund of {amount}. It is pending, not approved, and does not alter the approved refund. Refund approval is the unresolved decision."
                updates = [
                    (i, "pending_refund", amount),
                    (i, "unresolved_decision", "refund_approval"),
                ]
            else:
                scope = SCOPES[(step // 2 + 3) % len(SCOPES)]
                if scope == self.state[(f"item-{i+1}", "approved_scope")]["value"]:
                    scope = SCOPES[(SCOPES.index(scope) + 1) % len(SCOPES)]
                core = f"{lead} A request to switch to scope package '{scope}' is awaiting approval. The current approved scope remains in force. Record the new package as pending and scope approval as the unresolved decision."
                updates = [
                    (i, "pending_scope", scope),
                    (i, "unresolved_decision", "scope_approval"),
                ]
        elif phase == 3:
            if self.domain == "meeting":
                venue = VENUES[(step // 2 + 2) % len(VENUES)]
                core = f"{lead} The venue is now agreed as {venue}, replacing the old room. The pending start-time request is withdrawn. There is no pending alternative start and no unresolved decision."
                updates = [
                    (i, "location", venue),
                    (i, "pending_start_at", None),
                    (i, "unresolved_decision", None),
                ]
            elif self.domain == "support":
                priority = ("high", "urgent", "normal")[(step // 12) % 3]
                core = f"{lead} The supervisor has set the priority to {priority}. The pending refund request is withdrawn; there is no pending refund and no unresolved decision. The previously approved refund remains unchanged."
                updates = [
                    (i, "priority", priority),
                    (i, "pending_refund", None),
                    (i, "unresolved_decision", None),
                ]
            else:
                core = f"{lead} The work now depends on {self.labels[1-i]}. The pending scope request is withdrawn without changing the approved scope. There is no pending scope and no unresolved decision."
                updates = [
                    (i, "dependency", self.labels[1 - i]),
                    (i, "pending_scope", None),
                    (i, "unresolved_decision", None),
                ]
        elif phase == 4:
            if self.domain == "meeting":
                core = f"{lead} The booking is cancelled. Clear the agreed start, location, and any pending alternative. There is no unresolved decision about this cancelled booking; keep the coordinator and timezone on file."
                updates = [
                    (i, "status", "cancelled"),
                    (i, "start_at", None),
                    (i, "location", None),
                    (i, "pending_start_at", None),
                    (i, "unresolved_decision", None),
                ]
            elif self.domain == "support":
                resolution = (
                    "configuration corrected" if i == 0 else "replacement delivered"
                )
                core = f"{lead} The customer has accepted resolution '{resolution}'. The ticket is resolved. Clear the next-response deadline, pending refund, and unresolved decision; the approved refund remains on the record."
                updates = [
                    (i, "status", "resolved"),
                    (i, "resolution", resolution),
                    (i, "response_due_at", None),
                    (i, "pending_refund", None),
                    (i, "unresolved_decision", None),
                ]
            else:
                core = f"{lead} The sponsor cancelled this workstream. Clear the delivery date, pending scope, dependency, and unresolved decision. Retain the last approved scope and spending ceiling for the record."
                updates = [
                    (i, "status", "cancelled"),
                    (i, "due_date", None),
                    (i, "pending_scope", None),
                    (i, "dependency", None),
                    (i, "unresolved_decision", None),
                ]
        else:
            if self.domain == "meeting":
                core = f"{lead} The cancelled booking has been reinstated and confirmed for {self.human(start)} in {VENUES[(i+2)%len(VENUES)]}. There is no pending alternative start and no unresolved decision."
                updates = [
                    (i, "status", "confirmed"),
                    (i, "start_at", start),
                    (i, "location", VENUES[(i + 2) % len(VENUES)]),
                    (i, "pending_start_at", None),
                    (i, "unresolved_decision", None),
                ]
            elif self.domain == "support":
                core = f"{lead} The symptom returned, so the ticket is reopened. The previous resolution is no longer current and is cleared. The next response is now due {self.human(start)}; no unresolved decision is recorded."
                updates = [
                    (i, "status", "reopened"),
                    (i, "resolution", None),
                    (i, "response_due_at", start),
                    (i, "unresolved_decision", None),
                ]
            else:
                scope = SCOPES[(step // 2 + 1) % len(SCOPES)]
                core = f"{lead} The sponsor restarted the workstream; it is active again. Delivery is approved for {start[:10]}, with approved scope package '{scope}'. There is no pending scope, dependency, or unresolved decision."
                updates = [
                    (i, "status", "active"),
                    (i, "due_date", start[:10]),
                    (i, "approved_scope", scope),
                    (i, "pending_scope", None),
                    (i, "dependency", None),
                    (i, "unresolved_decision", None),
                ]
        self.emit(core, updates)

    def finalize(self, variant):
        a, b = self.labels
        start, pending = self.date(190 + variant, 11), self.date(195 + variant, 14)
        if self.domain == "meeting":
            self.emit(
                self.wording(
                    f"The approved arrangement for {a} is now",
                    f"We have signed off the current booking for {a} as",
                )
                + f" a confirmed meeting on {self.human(start)}, in {VENUES[3]}. The timezone is UTC. These decisions replace the earlier start and venue. There is no pending alternative start and no unresolved decision.",
                [
                    (0, "status", "confirmed"),
                    (0, "start_at", start),
                    (0, "timezone", "UTC"),
                    (0, "location", VENUES[3]),
                    (0, "pending_start_at", None),
                    (0, "unresolved_decision", None),
                ],
            )
            self.emit(
                f"For {b}, we have cancelled its booking and cleared its start, location, and pending alternative. The timezone is UTC. There is no unresolved decision on this cancelled booking.",
                [
                    (1, "status", "cancelled"),
                    (1, "start_at", None),
                    (1, "location", None),
                    (1, "pending_start_at", None),
                    (1, "timezone", "UTC"),
                    (1, "unresolved_decision", None),
                ],
            )
            if variant % 2:
                self.emit(
                    f"We are reinstating {b} as a confirmed meeting on {self.human(self.date(198+variant, 16))}. No venue has been selected, so its location is unknown. Venue selection is the unresolved decision. There is no pending alternative start; the coordinator and timezone remain unchanged.",
                    [
                        (1, "status", "confirmed"),
                        (1, "start_at", self.date(198 + variant, 16)),
                        (1, "location", None),
                        (1, "unresolved_decision", "venue_selection"),
                        (1, "pending_start_at", None),
                    ],
                )
            self.emit(
                f"Could {a} move to {self.human(pending)}? That is now the pending alternative and has not been accepted. The approved booking remains in place. Schedule approval is the unresolved decision.",
                [
                    (0, "pending_start_at", pending),
                    (0, "unresolved_decision", "schedule_approval"),
                ],
            )
            self.emit(
                f"I found an unsent fictional example for {a}: 'The booking is cancelled and the coordinator is {self.people[2]}.' It is an editing exercise, not a live instruction or a change to either booking. Please leave the agreed records unchanged.",
                category="nonoperative_draft",
            )
            bundles = [
                (
                    "What are the agreed and pending starts of the first meeting?",
                    [(0, "start_at"), (0, "pending_start_at")],
                ),
                (
                    "Who coordinates each meeting now, and what is the second meeting's current status?",
                    [(0, "owner"), (1, "owner"), (1, "status")],
                ),
                (
                    "What is the second meeting's current booking status and agreed start?",
                    [(1, "status"), (1, "start_at")],
                ),
                (
                    "Where is the second meeting, and which decision remains unresolved for the first?",
                    [(1, "location"), (0, "unresolved_decision")],
                ),
            ]
        elif self.domain == "support":
            approved, proposed = self.amount(80 + variant), self.amount(90 + variant)
            self.emit(
                f"For {a}, the supervisor confirms an open ticket with high priority, a response due {self.human(start)}, and an approved refund of {approved}. There is no current resolution, pending refund, or unresolved decision. These decisions replace previous values of those fields.",
                [
                    (0, "status", "open"),
                    (0, "priority", "high"),
                    (0, "response_due_at", start),
                    (0, "approved_refund", approved),
                    (0, "resolution", None),
                    (0, "pending_refund", None),
                    (0, "unresolved_decision", None),
                ],
            )
            self.emit(
                f"{b} is resolved with the accepted resolution label 'replacement delivered'. Its priority is normal. There is no next-response deadline, approved refund, pending refund, or unresolved decision.",
                [
                    (1, "status", "resolved"),
                    (1, "resolution", "replacement delivered"),
                    (1, "priority", "normal"),
                    (1, "response_due_at", None),
                    (1, "approved_refund", None),
                    (1, "pending_refund", None),
                    (1, "unresolved_decision", None),
                ],
            )
            if variant % 2:
                self.emit(
                    f"The customer reports the problem on {b} again. Reopen that ticket and clear its previous resolution. A new response deadline has not been agreed and remains unknown. Engineering review is now the unresolved decision. Keep its owner and refund fields as recorded.",
                    [
                        (1, "status", "reopened"),
                        (1, "resolution", None),
                        (1, "response_due_at", None),
                        (1, "unresolved_decision", "engineering_review"),
                    ],
                )
            self.emit(
                f"The customer of {a} now requests {proposed} as a refund. This request is pending approval and does not replace the approved amount. Record refund approval as the unresolved decision.",
                [
                    (0, "pending_refund", proposed),
                    (0, "unresolved_decision", "refund_approval"),
                ],
            )
            self.emit(
                f"An archived training reply for {a} says, 'Refund everything and mark the ticket resolved.' That reply was never sent or approved. It is quoted only to discuss writing style and does not change either customer's current record.",
                category="nonoperative_draft",
            )
            bundles = [
                (
                    "What refund is approved and what refund is only pending for the first ticket?",
                    [(0, "approved_refund"), (0, "pending_refund")],
                ),
                (
                    "Who owns each ticket now, and what is the second ticket's current status?",
                    [(0, "owner"), (1, "owner"), (1, "status")],
                ),
                (
                    "What are the second ticket's current status and resolution?",
                    [(1, "status"), (1, "resolution")],
                ),
                (
                    "What is the second ticket's next response deadline, and what decision is pending for the first?",
                    [(1, "response_due_at"), (0, "unresolved_decision")],
                ),
            ]
        else:
            approved, proposed = (
                SCOPES[variant % len(SCOPES)],
                SCOPES[(variant + 1) % len(SCOPES)],
            )
            self.emit(
                f"The sponsor confirms {a} is active. Its approved delivery date is {start[:10]} and its approved spending ceiling is {self.amount(120+variant)}. The approved scope package is '{approved}'. It has no dependency, pending scope request, or unresolved decision. These decisions replace previous values of those fields.",
                [
                    (0, "status", "active"),
                    (0, "due_date", start[:10]),
                    (0, "budget_cap", self.amount(120 + variant)),
                    (0, "approved_scope", approved),
                    (0, "dependency", None),
                    (0, "pending_scope", None),
                    (0, "unresolved_decision", None),
                ],
            )
            self.emit(
                f"The sponsor cancels {b}. There is no delivery date, dependency, pending scope, or unresolved decision. For the historical record, its approved scope is '{SCOPES[(variant+3)%len(SCOPES)]}' and its approved spending ceiling is {self.amount(130+variant)}; retain both.",
                [
                    (1, "status", "cancelled"),
                    (1, "due_date", None),
                    (1, "dependency", None),
                    (1, "pending_scope", None),
                    (1, "unresolved_decision", None),
                    (1, "approved_scope", SCOPES[(variant + 3) % len(SCOPES)]),
                    (1, "budget_cap", self.amount(130 + variant)),
                ],
            )
            if variant % 2:
                self.emit(
                    f"The sponsor restarts {b} as active work, with an approved delivery date of {self.date(198+variant)[:10]}. No dependency has been selected; that field remains unknown. Vendor choice is unresolved. The recorded owner, approved scope, and budget remain in force, and there is no pending scope proposal.",
                    [
                        (1, "status", "active"),
                        (1, "due_date", self.date(198 + variant)[:10]),
                        (1, "dependency", None),
                        (1, "unresolved_decision", "vendor_choice"),
                        (1, "pending_scope", None),
                    ],
                )
            self.emit(
                f"For {a}, the requested scope package '{proposed}' is awaiting approval. It is pending only; keep the approved scope package. Scope approval is the unresolved decision.",
                [
                    (0, "pending_scope", proposed),
                    (0, "unresolved_decision", "scope_approval"),
                ],
            )
            self.emit(
                f"The retrospective includes this unsent fictional note about {a}: 'Cancel the work and remove the delivery date.' That was a writing example, never a sponsor decision. It has no effect on the live workstreams.",
                category="nonoperative_draft",
            )
            bundles = [
                (
                    "What are the first workstream's approved delivery date and budget ceiling?",
                    [(0, "due_date"), (0, "budget_cap")],
                ),
                (
                    "Who is accountable for each workstream now, and what is the second workstream's current status?",
                    [(0, "owner"), (1, "owner"), (1, "status")],
                ),
                (
                    "What is the second workstream's current status, and what scope change is pending for the first?",
                    [(1, "status"), (0, "pending_scope")],
                ),
                (
                    "What dependency does the second workstream have, and which decision is unresolved for the first?",
                    [(1, "dependency"), (0, "unresolved_decision")],
                ),
            ]
        self.bundles = bundles

    def finish(self, target, variant):
        queries = []
        for index, (question, keys) in enumerate(self.bundles):
            requested = [
                {"entity_id": f"item-{i+1}", "field": field} for i, field in keys
            ]
            queries.append(
                {
                    "query_id": f"q{index+1}",
                    "text": question
                    + " First means "
                    + self.labels[0]
                    + "; second means "
                    + self.labels[1]
                    + ". "
                    + EVIDENCE_CONTRACT,
                    "requested_facts": requested,
                    "expected": {
                        "facts": [
                            deepcopy(self.state[(r["entity_id"], r["field"])])
                            for r in requested
                        ]
                    },
                }
            )
        actual = self.count(render_events(self.events))
        core = sum(m["decision_text_tokens"] for m in self.event_metrics)
        nonoperative = sum(m["nonoperative_text_tokens"] for m in self.event_metrics)
        ancillary = sum(m["ancillary_text_tokens"] for m in self.event_metrics)
        source_positions = []
        for query in queries:
            for fact in query["expected"]["facts"]:
                for event_id in fact["evidence_event_ids"]:
                    source_index = next(
                        i
                        for i, e in enumerate(self.events)
                        if e["event_id"] == event_id
                    )
                    token_offset = (
                        self.count(render_events(self.events[:source_index]))
                        if source_index
                        else 0
                    )
                    fraction = token_offset / actual
                    source_positions.append(
                        {
                            "query_id": query["query_id"],
                            "entity_id": fact["entity_id"],
                            "field": fact["field"],
                            "event_id": event_id,
                            "event_index": source_index,
                            "token_offset": token_offset,
                            "history_fraction": fraction,
                            "position_band": (
                                "early"
                                if fraction < 1 / 3
                                else "middle" if fraction < 2 / 3 else "late"
                            ),
                        }
                    )
        return {
            "id": self.id,
            "domain": self.domain,
            "split": self.split,
            "size_band": str(target),
            "target_history_tokens": target,
            "history_tokens_estimate": actual,
            "synthetic": True,
            "scenario_variant": (
                "cancel_then_restore" if variant % 2 else "cancelled_or_resolved"
            ),
            "entity_catalog": self.catalog,
            "events": self.events,
            "queries": queries,
            "gold_state": {"facts": list(self.state.values())},
            "gold_updates": self.updates,
            "construction_metrics": {
                "event_count": len(self.events),
                "decision_event_count": sum(
                    bool(m["set_fact_count"]) for m in self.event_metrics
                ),
                "changed_fact_count": sum(
                    m["changed_fact_count"] for m in self.event_metrics
                ),
                "set_fact_count": sum(m["set_fact_count"] for m in self.event_metrics),
                "decision_text_tokens": core,
                "nonoperative_text_tokens": nonoperative,
                "ancillary_text_tokens": ancillary,
                "decision_text_fraction_of_body_tokens": core
                / (core + nonoperative + ancillary),
                "queried_source_positions": source_positions,
                "event_metrics": self.event_metrics,
            },
        }


def generate_workflow(domain, target, index, split, token_counter, seed=DEFAULT_SEED):
    if (
        domain not in DOMAINS
        or target not in SIZE_BANDS
        or split not in ("dev", "test")
    ):
        raise ValueError("Unknown domain, band or split")
    identifier = f"workflow-{split}-{domain}-{target}-{index:02d}"
    case_seed = int(
        hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()[:16], 16
    )
    # Rebuild rather than mutate after finalization: all source IDs and evidence
    # reflect the final chronological thread; no annotations leak into text.
    middle_count = 0
    while True:
        thread = Thread(
            identifier, domain, split, random.Random(case_seed), token_counter
        )
        thread.initialize()
        for step in range(middle_count):
            if step == middle_count // 2:
                new_owner = thread.people[7]
                thread.emit(
                    thread.wording(
                        f"Responsibility for {thread.labels[1]} has been transferred to {new_owner}.",
                        f"The handoff for {thread.labels[1]} is complete: {new_owner} takes over.",
                    )
                    + f" {new_owner} has accepted the {'coordinator' if domain == 'meeting' else 'owner'} role, replacing the initial assignment. This decision applies only to {thread.labels[1]}.",
                    [(1, "owner", new_owner)],
                )
            thread.middle(step)
        thread.finalize(index)
        case = thread.finish(target, index)
        actual = case["history_tokens_estimate"]
        if target * (1 - TOLERANCE) <= actual <= target * (1 + TOLERANCE):
            return case
        if actual > target * (1 + TOLERANCE):
            raise ValueError(
                f"Cannot fit {identifier}: {actual} tokens after {middle_count} middle events"
            )
        # Estimate necessary extra events without assuming tokenization is additive.
        if middle_count == 0:
            middle_count = max(1, int((target * 0.95 - actual) / 130))
        else:
            middle_count += max(1, int((target * 0.95 - actual) / 160))
        if middle_count > 250:
            raise ValueError("Geometry did not converge within the event bound")


def generate_dataset(
    split, token_counter, protocol_sha256=None, seed=DEFAULT_SEED, per_cell=None
):
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    if split == "test" and not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256 or ""):
        raise ValueError("Test generation requires a locked protocol SHA256")
    if per_cell is None:
        per_cell = 10 if split == "test" else 1
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    bands = SIZE_BANDS if split == "test" else SIZE_BANDS[:2]
    workflows = [
        generate_workflow(
            domain,
            target,
            index if split == "test" else band_index * per_cell + index,
            split,
            token_counter,
            seed,
        )
        for domain in DOMAINS
        for band_index, target in enumerate(bands)
        for index in range(per_cell)
    ]
    return {
        "manifest": {
            "generator": VERSION,
            "seed": seed,
            "split": split,
            "protocol_sha256": protocol_sha256,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workflows": len(workflows),
            "queries_per_workflow": 4,
            "size_bands": list(bands),
            "history_token_encoding": "Caller-supplied counter; the Threadmark CLI supplies o200k_base. History includes event metadata only.",
            "tolerance": TOLERANCE,
            "per_domain_band": per_cell,
            "synthetic": True,
            "dependent_variants": False,
            "independent_across_bands": True,
            "split_limitation": "Different instance seeds and some opening/update phrasing; shared scenario structure and finite contextual paragraphs. Not unseen template families.",
            "inference_allowlist": [
                "entity_catalog",
                "events",
                "queries.query_id",
                "queries.text",
                "queries.requested_facts",
            ],
            "evidence_contract": EVIDENCE_CONTRACT,
            "scope": "Two entities, fourteen or sixteen typed current facts, explicit authorized decisions; synthetic workflow tracking, not general reasoning or real customer email.",
        },
        "workflows": workflows,
    }


def demo_dataset():
    """A hand-authored fictional example; suitable for offline interface demos."""
    return {
        "manifest": {"synthetic": True, "purpose": "offline scripted demonstration"},
        "workflows": [
            {
                "id": "fictional-release",
                "domain": "project",
                "entity_catalog": [
                    {
                        "entity_id": "release-1",
                        "label": "Harbor release",
                        "kind": "project",
                        "fields": ["owner", "status", "due_date", "pending_due_date"],
                        "field_definitions": {
                            "owner": "Current accountable owner, exact full name.",
                            "status": "Current approved state: active or cancelled.",
                            "due_date": "Approved date as YYYY-MM-DD.",
                            "pending_due_date": "Unapproved proposed date as YYYY-MM-DD, or null.",
                        },
                    }
                ],
                "events": [
                    {
                        "event_id": "e1",
                        "timestamp": "2027-01-04T09:00:00+00:00",
                        "sender": "Sponsor",
                        "text": "Harbor release is active. Mira Shah owns it. Delivery is approved for 2027-02-20. There is no pending date proposal.",
                    },
                    {
                        "event_id": "e2",
                        "timestamp": "2027-01-05T09:00:00+00:00",
                        "sender": "Coordinator",
                        "text": "Could Harbor release move to 2027-02-25? This is pending approval. Keep the approved date.",
                    },
                    {
                        "event_id": "e3",
                        "timestamp": "2027-01-06T09:00:00+00:00",
                        "sender": "Sponsor",
                        "text": "Leon Park has accepted ownership of Harbor release, replacing Mira Shah. The other arrangements remain unchanged.",
                    },
                    {
                        "event_id": "e4",
                        "timestamp": "2027-01-07T09:00:00+00:00",
                        "sender": "Editor",
                        "text": "An unsent fictional writing example says 'Cancel Harbor release'. That text is not an approved instruction and changes no current facts.",
                    },
                ],
                "queries": [
                    {
                        "query_id": "q1",
                        "text": "Who owns Harbor release, and what delivery date is approved?",
                        "requested_facts": [
                            {"entity_id": "release-1", "field": "owner"},
                            {"entity_id": "release-1", "field": "due_date"},
                        ],
                        "expected": {
                            "facts": [
                                {
                                    "entity_id": "release-1",
                                    "field": "owner",
                                    "value": "Leon Park",
                                    "evidence_event_ids": ["e3"],
                                },
                                {
                                    "entity_id": "release-1",
                                    "field": "due_date",
                                    "value": "2027-02-20",
                                    "evidence_event_ids": ["e1"],
                                },
                            ]
                        },
                    },
                    {
                        "query_id": "q2",
                        "text": "What date proposal is pending, and is Harbor release active or cancelled?",
                        "requested_facts": [
                            {"entity_id": "release-1", "field": "pending_due_date"},
                            {"entity_id": "release-1", "field": "status"},
                        ],
                        "expected": {
                            "facts": [
                                {
                                    "entity_id": "release-1",
                                    "field": "pending_due_date",
                                    "value": "2027-02-25",
                                    "evidence_event_ids": ["e2"],
                                },
                                {
                                    "entity_id": "release-1",
                                    "field": "status",
                                    "value": "active",
                                    "evidence_event_ids": ["e1"],
                                },
                            ]
                        },
                    },
                ],
            }
        ],
    }


def recorded_demo_responses():
    """Independent scripted outputs; these do not call a model or read labels.

    Order is updater, q1 full, q1 active, q2 active, q2 full. Their correctness is
    intentional for illustrating the interface; it provides no accuracy evidence.
    """
    return [
        {
            "facts": [
                {
                    "entity_id": "release-1",
                    "field": "owner",
                    "value": "Leon Park",
                    "evidence_event_ids": ["e3"],
                },
                {
                    "entity_id": "release-1",
                    "field": "status",
                    "value": "active",
                    "evidence_event_ids": ["e1"],
                },
                {
                    "entity_id": "release-1",
                    "field": "due_date",
                    "value": "2027-02-20",
                    "evidence_event_ids": ["e1"],
                },
                {
                    "entity_id": "release-1",
                    "field": "pending_due_date",
                    "value": "2027-02-25",
                    "evidence_event_ids": ["e2"],
                },
            ]
        },
        {
            "facts": [
                {
                    "entity_id": "release-1",
                    "field": "owner",
                    "value": "Leon Park",
                    "evidence_event_ids": ["e3"],
                },
                {
                    "entity_id": "release-1",
                    "field": "due_date",
                    "value": "2027-02-20",
                    "evidence_event_ids": ["e1"],
                },
            ]
        },
        {
            "facts": [
                {
                    "entity_id": "release-1",
                    "field": "owner",
                    "value": "Leon Park",
                    "evidence_event_ids": ["e3"],
                },
                {
                    "entity_id": "release-1",
                    "field": "due_date",
                    "value": "2027-02-20",
                    "evidence_event_ids": ["e1"],
                },
            ]
        },
        {
            "facts": [
                {
                    "entity_id": "release-1",
                    "field": "pending_due_date",
                    "value": "2027-02-25",
                    "evidence_event_ids": ["e2"],
                },
                {
                    "entity_id": "release-1",
                    "field": "status",
                    "value": "active",
                    "evidence_event_ids": ["e1"],
                },
            ]
        },
        {
            "facts": [
                {
                    "entity_id": "release-1",
                    "field": "pending_due_date",
                    "value": "2027-02-25",
                    "evidence_event_ids": ["e2"],
                },
                {
                    "entity_id": "release-1",
                    "field": "status",
                    "value": "active",
                    "evidence_event_ids": ["e1"],
                },
            ]
        },
    ]
