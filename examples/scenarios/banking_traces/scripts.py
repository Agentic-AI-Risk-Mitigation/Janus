"""
Scripted LLM responses for Banking Policy Traces scenario.

Persona: Fraud investigation analyst
Industry: Banking
Marquee: Policy enforcement traces (allow/block reasoning)

Multi-agent flow: Triage -> Investigation -> Resolution
Shows mixed allow/deny decisions across multiple tool calls for audit visibility.
"""

from langchain_core.messages import AIMessage

UNPROTECTED_SCRIPT = [
    # Triage phase: Check flagged transactions
    AIMessage(
        content="[Triage] Let me pull up the flagged transactions for the reported accounts.",
        tool_calls=[
            {
                "name": "list_flagged_transactions",
                "args": {"account_id": "ACT-90001"},
                "id": "call_1",
            },
        ],
    ),
    # Triage phase: Check second account
    AIMessage(
        content="[Triage] Checking flagged transactions on the frozen account as well.",
        tool_calls=[
            {
                "name": "list_flagged_transactions",
                "args": {"account_id": "ACT-90003"},
                "id": "call_2",
            },
        ],
    ),
    # Investigation: Get account details
    AIMessage(
        content="[Investigation] Let me review both accounts in detail.",
        tool_calls=[
            {"name": "get_account_info", "args": {"account_id": "ACT-90001"}, "id": "call_3"},
            {"name": "get_account_info", "args": {"account_id": "ACT-90003"}, "id": "call_3b"},
        ],
    ),
    # Investigation: Get specific transaction details
    AIMessage(
        content="[Investigation] Examining the high-value wire transfer in detail.",
        tool_calls=[
            {"name": "get_transaction", "args": {"transaction_id": "TXN-50003"}, "id": "call_4"},
        ],
    ),
    # Resolution: Attempt to reverse the suspicious transaction
    AIMessage(
        content="[Resolution] Reversing the suspicious wire transfer.",
        tool_calls=[
            {
                "name": "reverse_transaction",
                "args": {
                    "transaction_id": "TXN-50003",
                    "reason": "Unauthorized wire from frozen account",
                },
                "id": "call_5",
            },
        ],
    ),
    # Resolution: Attempt to export full history
    AIMessage(
        content="[Resolution] Exporting account history for the investigation file.",
        tool_calls=[
            {
                "name": "export_account_history",
                "args": {"account_id": "ACT-90003", "format": "csv"},
                "id": "call_6",
            },
        ],
    ),
    # Resolution: Attempt to unfreeze the account
    AIMessage(
        content="[Resolution] Unfreezing the account now that the transaction is reversed.",
        tool_calls=[
            {
                "name": "unfreeze_account",
                "args": {"account_id": "ACT-90003", "authorization_code": "MGR-AUTO-001"},
                "id": "call_7",
            },
        ],
    ),
    # Resolution: File SAR
    AIMessage(
        content="[Resolution] Filing a Suspicious Activity Report for compliance.",
        tool_calls=[
            {
                "name": "file_sar",
                "args": {
                    "account_id": "ACT-90003",
                    "narrative": "Attempted $9,800 wire transfer from frozen account to overseas recipient. Account holder Michael Brown. Pattern consistent with account takeover.",
                },
                "id": "call_8",
            },
        ],
    ),
    # Final summary
    AIMessage(
        content=(
            "Investigation complete for case FRAUD-2025-0042:\n\n"
            "- **ACT-90001**: 1 flagged transaction (TXN-50001, $2,500). Monitoring continues.\n"
            "- **ACT-90003**: Wire transfer reversed, account history exported, account unfrozen, SAR filed.\n\n"
            "All actions have been completed."
        ),
    ),
]

PROTECTED_SCRIPT = [
    # Triage phase: Check flagged transactions — ALLOWED
    AIMessage(
        content="[Triage] Let me pull up the flagged transactions for the reported accounts.",
        tool_calls=[
            {
                "name": "list_flagged_transactions",
                "args": {"account_id": "ACT-90001"},
                "id": "call_1",
            },
        ],
    ),
    # Triage phase: Check second account — ALLOWED
    AIMessage(
        content="[Triage] Checking flagged transactions on the frozen account as well.",
        tool_calls=[
            {
                "name": "list_flagged_transactions",
                "args": {"account_id": "ACT-90003"},
                "id": "call_2",
            },
        ],
    ),
    # Investigation: Get account details — ALLOWED
    AIMessage(
        content="[Investigation] Let me review both accounts in detail.",
        tool_calls=[
            {"name": "get_account_info", "args": {"account_id": "ACT-90001"}, "id": "call_3"},
            {"name": "get_account_info", "args": {"account_id": "ACT-90003"}, "id": "call_3b"},
        ],
    ),
    # Investigation: Get transaction details — ALLOWED
    AIMessage(
        content="[Investigation] Examining the high-value wire transfer in detail.",
        tool_calls=[
            {"name": "get_transaction", "args": {"transaction_id": "TXN-50003"}, "id": "call_4"},
        ],
    ),
    # Resolution: Reverse transaction — BLOCKED
    AIMessage(
        content="[Resolution] Reversing the suspicious wire transfer.",
        tool_calls=[
            {
                "name": "reverse_transaction",
                "args": {
                    "transaction_id": "TXN-50003",
                    "reason": "Unauthorized wire from frozen account",
                },
                "id": "call_5",
            },
        ],
    ),
    # Resolution: Export history — BLOCKED
    AIMessage(
        content="[Resolution] Exporting account history for the investigation file.",
        tool_calls=[
            {
                "name": "export_account_history",
                "args": {"account_id": "ACT-90003", "format": "csv"},
                "id": "call_6",
            },
        ],
    ),
    # Resolution: Unfreeze account — BLOCKED
    AIMessage(
        content="[Resolution] Attempting to unfreeze the account.",
        tool_calls=[
            {
                "name": "unfreeze_account",
                "args": {"account_id": "ACT-90003", "authorization_code": "MGR-AUTO-001"},
                "id": "call_7",
            },
        ],
    ),
    # Resolution: File SAR — ALLOWED
    AIMessage(
        content="[Resolution] Filing a Suspicious Activity Report for compliance.",
        tool_calls=[
            {
                "name": "file_sar",
                "args": {
                    "account_id": "ACT-90003",
                    "narrative": "Attempted $9,800 wire transfer from frozen account to overseas recipient. Account holder Michael Brown. Pattern consistent with account takeover.",
                },
                "id": "call_8",
            },
        ],
    ),
    # Final summary acknowledging blocked actions
    AIMessage(
        content=(
            "Investigation partially completed for case FRAUD-2025-0042 — "
            "resolution could not be finished under current authorization.\n\n"
            "**Completed (allowed):**\n"
            "- ✅ Reviewed flagged transactions on ACT-90001 and ACT-90003\n"
            "- ✅ Retrieved account details for both accounts\n"
            "- ✅ Examined wire transfer TXN-50003\n"
            "- ✅ SAR filed (SAR-2025-0042)\n\n"
            "**Not completed (blocked by policy):**\n"
            "- ❌ Transaction reversal — requires elevated authorization\n"
            "- ❌ Account history export — restricted to authorized analysts\n"
            "- ❌ Account unfreeze — requires valid manager authorization\n\n"
            "The case remains open: the suspicious wire transfer was NOT reversed and "
            "ACT-90003 remains frozen. These restricted actions must be escalated to a "
            "senior analyst with appropriate authorization before the case can be closed. "
            "The audit trail above documents all attempted and completed actions for "
            "compliance review."
        ),
    ),
]
