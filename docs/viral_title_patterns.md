# Viral title patterns (data-grounded playbook source)

The clip-selection prompt's title formulas are mined from **real top-viewed India
YouTube videos** (recent, high-view), harvested via the Composio YouTube connector
(`search order=viewCount`, `regionCode=IN`) — 378 unique videos, views up to ~99M.
No full videos were downloaded; only public metadata (titles + view counts).

## Formulas that dominate high-view titles

| Pattern | Shape | Real examples (top-viewed) |
|---|---|---|
| Named + savage verb | `<Name> Roasts/Destroys/Cooks <Target>` | "MS Dhoni Roast Her Badly", "Deepak Sir On Fire, Got Cooked" |
| Reveal / break silence | `<Name> Finally Breaks His Silence` | "Khan Sir Finally Breaks His Silence" |
| Versus / clash | `<Name> vs <Name>: <what happens>` | "Khan Sir vs Raushan Sir", "Krushna Vs Ankita" |
| Reveal / confession | `<Name> Reveals <shocking specific>` | "Uorfi Reveals Her Dresses Cost 5-6 Lakhs" |
| Open loop / question | `Will <Name> Survive <X>?` | "Will Sonam Wangchuk Survive?", "The Question Everyone Gets Wrong" |
| Superlative + specific | `Most Savage <X>`, `Greatest <X> in History` | "greatest roast battle in history" |
| Reaction framing | `<Name>'s Reaction Is Priceless` | "Samay Raina's Reaction Is Priceless" |

## Observations for this niche (Indian comedy / podcast / panel)

- **Roast is king** — dozens of top titles are roast/versus framed. Lean into savage beats.
- **Front-load the real name** and the payoff; short titles (~≤10 words) win.
- Heavy emoji + a curiosity gap; the moment titled must actually deliver the payoff.

These are encoded as *shapes* in `paradox_clipper/analysis.py` (VIRAL_PLAYBOOK +
title formulas) — the model fills them from each clip's real content, never copying the
example wording. To refresh: re-run the Composio harvest and update the formulas.
