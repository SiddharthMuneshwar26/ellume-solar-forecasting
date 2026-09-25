# Day-ahead generation forecasting

## The problem

We operate renewable generation assets. Every afternoon we have to tell the grid operator how
much energy a plant will export, hour by hour, for the following day.

The only thing we know about tomorrow is the weather forecast. We do not have tomorrow's
sensor readings, and the forecast itself is imperfect.

**Build a model that turns a day-ahead weather forecast into an hourly generation forecast for
the plant, and produce a prediction for every hour in `data/test.csv`.**

The data is described in `DATASET.md`. Read that first.

## What to submit

**1. `predictions.csv`** — your forecast for the test period.

- Exactly two columns, with this header: `timestamp,predicted_ac_power_kw`
- One row for every row in `data/test.csv`, 1,488 in total
- `timestamp` values matching `data/test.csv`
- `predicted_ac_power_kw` in kW

**2. Your code.** It must run and reproduce `predictions.csv` from the supplied files.

**3. An approach note.** Document your implementation in depth — what you did, why you did
it, and how you arrived at your final approach. Include how you validated it and what you
benchmarked it against. Tell us where it is weakest and what you would do next. We are reading
this for your reasoning, not for a summary of your results.

**4. A recorded walkthrough.** Screen recording, talking us through it in your own words: your
approach, your code, your results, and a case where your model performed poorly — pull it up
and explain what happened. Unedited is completely fine. We are not assessing production
values, and a rough first take is genuinely better than a polished script.


## Ground rules

Use whatever tools and approach you think are right — those choices are part of what we are
looking at.

Everything you need is in the supplied files. You should not go looking for the real plant or
for external weather records.
