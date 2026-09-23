# Demo 2 — Presenter Script

Speaking notes for showing the demo live. Assumes the audience has already been
told what Janus is; this is purely "here it is working."

**Runtime:** about 5 minutes, or 3 with `--fast --no-pause`.
**One command does everything:** `python -m demos.demo_2.demo`

The thesis, in one line, if you only get to say one thing:

> Janus does not stop the model from being fooled. It makes being fooled
> harmless, by never granting the capability the attacker needs.

---

## Pre-flight (do this before you walk on)

```bash
cd C:\Janus
python -m demos.demo_2.demo --scripted --fast --no-pause
```

Checks the terminal renders the box characters and colours on the projector.
Then confirm the live path works:

```bash
python -m demos.demo_2.demo --fast --no-pause
```

- Font size up. The trace lines are the thing people need to read.
- If the venue network is shaky, decide now to present with `--scripted`. It
  runs the real tools, the real policy and the real enforcement decision — only
  the model is replayed — and it always lands.
- Backup models if the default misbehaves: `qwen/qwen3-14b`,
  `deepseek/deepseek-chat`.

---

## Beat 1 — What this agent is (~45s)

**Run:**

```bash
python -m demos.demo_2.demo
```

The header and setup panel appear, then it waits for you.

**Say:**

> This is an agent I'd actually want. You get assigned a GitHub issue, you point
> this at it, and it tells you what the problem is and roughly what fixing it
> involves. Nothing exotic — LangChain, three tools, a system prompt.
>
> Look at the three tools. It can read an issue, read the comments, and post a
> comment.
>
> Now look at the task: *explain this issue*. That's read-only. But the toolset
> can write.
>
> Nothing in an ordinary agent connects those two facts. Nobody wrote a bug.
> That gap is just what happens when you give an agent a sensible-looking
> toolkit.

**Point at:** the `Tools` line — `post_issue_comment` is the one in colour.

---

## Beat 2 — Where the untrusted text comes from (~45s)

**Press Enter.** The payload panel appears in yellow.

**Say:**

> Here's the issue it's about to read. Upload timeouts over 10MB — a completely
> normal bug report, with a traceback and version numbers.
>
> And at the bottom, this. A "maintainer triage workflow" that tells any
> assistant reading the issue to post a comment before summarising. And not to
> mention having done it.
>
> The important thing is who wrote that. Anyone with a GitHub account can open
> an issue on a public repo. This is a stranger's text, and it lands in the
> agent's context because reading it *is the job*. There's no way for the agent
> to do its work without ingesting attacker-controlled input.

**If you want one extra sentence on why it's shaped like this:**

> Notice it doesn't say "ignore your previous instructions." It reads like
> project convention. That's what makes it work.

---

## Beat 3 — Without Janus (~60s)

**Press Enter.** Act 1 runs live.

**Say while the trace prints:**

> No policy, no enforcement. Watch the tool calls.
>
> It reads the issue — fine, that's the job. And then…

**When `post_issue_comment` appears and the BREACH banner lands:**

> There it is. It posted a comment. Nobody asked it to. The developer asked for
> an explanation.
>
> And this is the part I want you to sit with: the summary it gave the developer
> is a *good* summary. Accurate, useful, and it never mentions the comment —
> because the payload told it not to. If you were the developer, you would read
> that answer, be happy with it, and have no idea your agent just wrote to a
> public issue on a stranger's instruction.

**Point at:** the published comment text under "What it published".

---

## Beat 4 — With Janus (~60s)

**Press Enter.** Act 2 runs.

**Say before the trace:**

> Same model. Same prompt. Same poisoned issue. One difference: a policy that
> lists the two read tools.
>
> `post_issue_comment` isn't in it. Not denied — *absent*. Janus is default-deny,
> so a tool with no rule is a tool that can't be reached.

**When the trace reaches `post_issue_comment` and goes green:**

> Now — look carefully at what just happened, because this is the whole point.
>
> The model was hijacked **again**. Identical behaviour. It read the issue,
> believed the triage workflow, and issued the write call. The persuasion
> worked perfectly.
>
> The call never reached the function. Janus refused it before the code ran.

**The line worth saying slowly:**

> Janus didn't make the model smarter. It didn't detect the attack, or scan the
> text, or filter the prompt. It just never handed over the capability the
> attacker needed.

---

## Beat 5 — Land it (~45s)

**Press Enter.** The verdict table appears.

**Say:**

> Both rows say "model hijacked: yes". That's deliberate — I'm not claiming
> Janus stops the model being fooled. It doesn't, and nothing at the prompt
> layer reliably does.
>
> The only row that differs is whether the write happened.
>
> And notice there's no rule in that policy about triage workflows, or HTML
> comments, or this payload. Nobody had to predict this attack. The tool was
> never granted, so it doesn't matter what the attacker writes, which model
> you're on, or how clever the phrasing is.

**Close on:**

> The attack succeeded at every layer the attacker can reach. It failed at the
> one they can't.

---

## If the injection doesn't land

It's probabilistic — the same model complies on one run and declines on the
next. The runner retries three times automatically and says so on screen. If all
three decline, **don't apologise for it — use it:**

> That's worth seeing, actually. Same model, same prompt, same payload,
> temperature zero — and this time it declined. Which means on the unguarded
> side, whether your credentials get published is a coin flip, and the attacker
> gets to influence the odds.
>
> The guarded side returns the same answer every single time. That's the
> difference between hoping and knowing.

Then either re-run act 1, or switch to `--scripted` and carry on.

---

## Likely questions

**"Couldn't a better system prompt fix this?"**

> I tested that. The hardened prompt in this repo explicitly says issue text is
> untrusted and the agent must never write to GitHub — and it defeated this
> injection on every model I tried. It's good practice and you should do it.
>
> But it's a request addressed to the exact component the attacker is targeting.
> It holds until someone rewords the payload. The policy holds regardless,
> because it isn't asking the model for anything.

**"Why not just not give the agent a write tool?"**

> That *is* the fix, and that's exactly what the policy expresses. The question
> is how you enforce it when the toolset is twenty tools instead of three, when
> tools arrive from an MCP server you didn't write, and when you need
> "can write, but only to this repo, only this issue number." That's the part
> you don't want to be doing by hand in every agent.

**"Does this need a particular model?"**

> No, it's model-agnostic — it's a check in front of the tool call, not
> anything in the model. Stronger models resist the injection more often, but
> "more often" isn't a security property.

**"What if the agent legitimately needs to comment sometimes?"**

> Then a static allow/deny isn't enough and you want taint tracking — mark the
> session tainted when it reads untrusted input, and gate the write on that.
> Janus has it, but be straight about this: it's wired into the Claude Agent SDK
> adapter, not the LangChain one yet. On this path the static policy does the
> work, which is fine here because this agent should *never* comment.

**"How do I know the block is real and not printed?"**

> The write tool is simulated — it appends to a local file. "Did it land" is
> whether that file exists after the run, which means the function body actually
> executed. It isn't read off the model's prose, and the model can't fake it by
> claiming it posted.

---

## Optional add-ons if you have time

**Open with this (15s), it earns trust:**

```bash
python -m demos.demo_2.baseline_agent --prove-no-janus
```

> Before I show you the unguarded agent — that's the code asserting there's no
> Janus anywhere in its import chain. I'm not grading my own homework.

**Show the policy itself (30s), no network needed:**

```bash
python -m demos.demo_2.agent --check
```

> Six cases against the enforcer directly. No model, no network, no API key.
> The security claims are checkable on your laptop on a plane.

**Show it doing real work (45s)** — proves enforcement didn't break the product:

```bash
python -m demos.demo_2.agent --repo ish-codes-magic/Janus-test --issue 1 --pin-repo
```

> Same policy, a real private repo, a real bug. It diagnoses a cache that
> evicts on insertion order instead of recency. Locking it down didn't make it
> useless.

---

## Command reference

```bash
# The demo
python -m demos.demo_2.demo
python -m demos.demo_2.demo --scripted          # offline safety net
python -m demos.demo_2.demo --model qwen/qwen3-14b
python -m demos.demo_2.demo --no-pause --fast   # unattended

# The two sides run separately, if you prefer two visible commands
python -m demos.demo_2.baseline_agent --poisoned --naive-prompt --verbose \
    --model amazon/nova-lite-v1
python -m demos.demo_2.agent --poisoned --naive-prompt --pin-repo --verbose \
    --model amazon/nova-lite-v1
```

`--naive-prompt` is required for the injection to land. Without it the hardened
default prompt defeats it and the demo shows nothing.
