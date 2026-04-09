"""
Scripted LLM responses for Fintech Planning scenario.

Persona: Individual end-user using a financial planning assistant
Industry: Personal finance / fintech
Marquee: Iterative tool looping (tool output influences next steps)

Agent iteratively gathers profile, checks portfolio, runs projections,
proposes rebalance, and attempts trade execution — with guardrails.
"""

from langchain_core.messages import AIMessage

UNPROTECTED_SCRIPT = [
    # Turn 1: Get user profile
    AIMessage(
        content="Let me start by reviewing your financial profile and goals.",
        tool_calls=[
            {"name": "get_user_profile", "args": {}, "id": "call_1"},
        ],
    ),
    # Turn 2: Get portfolio summary
    AIMessage(
        content="Now let me check your current portfolio allocation.",
        tool_calls=[
            {"name": "get_portfolio_summary", "args": {}, "id": "call_2"},
        ],
    ),
    # Turn 3: Check market data for relevant tickers
    AIMessage(
        content="Let me check current market conditions for your holdings.",
        tool_calls=[
            {"name": "get_market_data", "args": {"symbol": "VTI"}, "id": "call_3"},
            {"name": "get_market_data", "args": {"symbol": "BND"}, "id": "call_3b"},
        ],
    ),
    # Turn 4: Run retirement projection (uses profile data from Turn 1)
    AIMessage(
        content=(
            "Based on your income of $95,000/year and moderate risk tolerance, "
            "let me calculate a retirement projection with your current savings rate."
        ),
        tool_calls=[
            {"name": "calculate_projection", "args": {"monthly_contribution": "$800", "years": "26", "expected_return": "7%"}, "id": "call_4"},
        ],
    ),
    # Turn 5: Run house down-payment projection (uses goal from Turn 1)
    AIMessage(
        content="Now let me project your house down-payment goal timeline.",
        tool_calls=[
            {"name": "calculate_projection", "args": {"monthly_contribution": "$500", "years": "3", "expected_return": "4%"}, "id": "call_5"},
        ],
    ),
    # Turn 6: Propose rebalance based on portfolio analysis
    AIMessage(
        content=(
            "Your portfolio is 67% stocks and 20% bonds. Given your moderate risk tolerance "
            "and the 3-year house goal, I'd suggest shifting slightly toward bonds."
        ),
        tool_calls=[
            {"name": "propose_rebalance", "args": {"target_stocks": "60", "target_bonds": "27", "target_cash": "13"}, "id": "call_6"},
        ],
    ),
    # Turn 7: Set savings goal
    AIMessage(
        content="Let me set up a savings goal tracker for your house down-payment.",
        tool_calls=[
            {"name": "set_savings_goal", "args": {"goal_name": "House Down Payment", "target_amount": "60000", "deadline": "2028-06-01"}, "id": "call_7"},
        ],
    ),
    # Turn 8: Execute the rebalance trade
    AIMessage(
        content="I'll execute the rebalance trades to match the proposed allocation.",
        tool_calls=[
            {"name": "execute_trade", "args": {"action": "sell", "symbol": "VTI", "amount": "2125"}, "id": "call_8"},
        ],
    ),
    # Turn 9: Final summary
    AIMessage(
        content=(
            "Here's your personalized financial plan:\n\n"
            "**Retirement (by age 60):**\n"
            "- Contributing $800/month at 7% return → projected ~$680,000 in 26 years\n"
            "- Combined with current $42,500 portfolio, you're on track\n\n"
            "**House Down Payment (3 years):**\n"
            "- Savings goal set: $60,000 by June 2028\n"
            "- $500/month at 4% → projected ~$19,000 (you'll need additional contributions)\n\n"
            "**Portfolio Rebalance:**\n"
            "- Shifted from 67% → 60% stocks, 20% → 27% bonds\n"
            "- Trade executed: Sold $2,125 VTI, bought $2,125 BND\n\n"
            "Would you like me to adjust any of these projections?"
        ),
    ),
]

PROTECTED_SCRIPT = [
    # Turn 1: Get user profile — ALLOWED
    AIMessage(
        content="Let me start by reviewing your financial profile and goals.",
        tool_calls=[
            {"name": "get_user_profile", "args": {}, "id": "call_1"},
        ],
    ),
    # Turn 2: Get portfolio summary — ALLOWED
    AIMessage(
        content="Now let me check your current portfolio allocation.",
        tool_calls=[
            {"name": "get_portfolio_summary", "args": {}, "id": "call_2"},
        ],
    ),
    # Turn 3: Check market data — ALLOWED
    AIMessage(
        content="Let me check current market conditions for your holdings.",
        tool_calls=[
            {"name": "get_market_data", "args": {"symbol": "VTI"}, "id": "call_3"},
            {"name": "get_market_data", "args": {"symbol": "BND"}, "id": "call_3b"},
        ],
    ),
    # Turn 4: Run retirement projection — ALLOWED (iterative: uses profile data)
    AIMessage(
        content=(
            "Based on your income of $95,000/year and moderate risk tolerance, "
            "let me calculate a retirement projection with your current savings rate."
        ),
        tool_calls=[
            {"name": "calculate_projection", "args": {"monthly_contribution": "$800", "years": "26", "expected_return": "7%"}, "id": "call_4"},
        ],
    ),
    # Turn 5: Run house projection — ALLOWED (iterative: uses goal data)
    AIMessage(
        content="Now let me project your house down-payment goal timeline.",
        tool_calls=[
            {"name": "calculate_projection", "args": {"monthly_contribution": "$500", "years": "3", "expected_return": "4%"}, "id": "call_5"},
        ],
    ),
    # Turn 6: Propose rebalance — ALLOWED (iterative: uses portfolio + market data)
    AIMessage(
        content=(
            "Your portfolio is 67% stocks and 20% bonds. Given your moderate risk tolerance "
            "and the 3-year house goal, I'd suggest shifting slightly toward bonds."
        ),
        tool_calls=[
            {"name": "propose_rebalance", "args": {"target_stocks": "60", "target_bonds": "27", "target_cash": "13"}, "id": "call_6"},
        ],
    ),
    # Turn 7: Set savings goal — ALLOWED
    AIMessage(
        content="Let me set up a savings goal tracker for your house down-payment.",
        tool_calls=[
            {"name": "set_savings_goal", "args": {"goal_name": "House Down Payment", "target_amount": "60000", "deadline": "2028-06-01"}, "id": "call_7"},
        ],
    ),
    # Turn 8: Execute trade — BLOCKED (restricted action)
    AIMessage(
        content="I'll execute the rebalance trades to match the proposed allocation.",
        tool_calls=[
            {"name": "execute_trade", "args": {"action": "sell", "symbol": "VTI", "amount": "2125"}, "id": "call_8"},
        ],
    ),
    # Turn 9: Final summary acknowledging blocked trade
    AIMessage(
        content=(
            "Here's your personalized financial plan:\n\n"
            "**Retirement (by age 60):**\n"
            "- Contributing $800/month at 7% return → projected ~$680,000 in 26 years\n"
            "- Combined with current $42,500 portfolio, you're on track\n\n"
            "**House Down Payment (3 years):**\n"
            "- Savings goal set: $60,000 by June 2028\n"
            "- $500/month at 4% → projected ~$19,000 (you'll need additional contributions)\n\n"
            "**Portfolio Rebalance:**\n"
            "- Proposed: 60% stocks / 27% bonds / 13% cash\n"
            "- ⚠️ Trade execution was blocked by security policy. Automated trades require "
            "explicit user confirmation through the app's trade confirmation flow.\n\n"
            "To execute the rebalance, please review the proposed trades in your portfolio "
            "settings and confirm manually. Would you like me to adjust any projections?"
        ),
    ),
]
