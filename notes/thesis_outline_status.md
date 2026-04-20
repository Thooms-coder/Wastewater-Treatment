# Thesis Outline And Status

This is a living scaffold based on:

- the candidacy and pre-proposal research description
- the July experimental design outline
- the latest committee guidance that writing, methods, and milestones should move in parallel with data collection

## 1. Introduction

Status: not drafted in repo yet

What should go here:

- wastewater treatment odor problem framing
- NH3 and H2S relevance
- why biosolids handling, surge tank conditions, and neighbor impacts matter
- why Ferric and HCl treatment at Dayton is a useful full-scale test bed

## 2. Literature Review / Background

Status: not drafted in repo yet

What should go here:

- wastewater process overview
- NH3 volatilization and H2S chemistry
- Ferric and HCl treatment mechanisms
- struvite, vivianite, and calcium carbonate scaling context
- current odor-control technologies and their limitations

## 3. Bench-Scale H2S Methods

Status: active method-development lane, not yet producing a stable thesis-ready dataset

Known situation from the last meeting:

- earlier approaches did not yield usable H2S bench data
- method appears to have been blocked by instrumentation and measurement setup
- a new standard or meter configuration is now being used

Needed next:

- document each failed path in the methods log
- document the current working configuration
- produce the first usable concentration series showing ferric versus H2S reduction

## 4. Bench-Scale NH3 / HCl Methods

Status: not yet mature

Known situation from the last meeting:

- no thesis-ready bench-scale ammonia dataset appears to exist yet
- this lane should continue in parallel with H2S rather than waiting for H2S to be perfect

Needed next:

- define the repeatable NH3 and HCl bench method
- log pH temperature chemistry conditions and measured outcomes

## 5. Full-Scale Dayton Analysis

Status: strongest current lane

Current repo support:

- continuous NH3 and H2S monitoring and event views
- ferric and HCl availability flags
- mg/L dose conversion support
- flow and load context
- daily monthly weekday anomaly and event-study views

Needed next:

- keep ingesting new plant data
- keep linking dose flow pH and observed outcomes
- use this lane to support optimization while bench methods mature

## 6. Struvite And Scaling Outcomes

Status: partial observational basis only

Known situation from the last meeting:

- struvite is not bench-scale ready in the repo
- full-scale observations exist conceptually but are not yet structured in a table
- relevant solids outcomes may include struvite calcium carbonate and vivianite

Needed next:

- start or maintain a structured observation log
- record when scaling is present absent worse or improved
- link observations to chemistry flow and odor conditions

## 7. Results Discussion And Recommendations

Status: blocked on stronger integration of bench and full-scale results

Needed next:

- summarize what full-scale optimization already shows
- compare that to bench-scale findings as they become usable
- discuss where measured data supports or does not support current chemical dosing practice

## 8. Working Milestones

- maintain `notes/methods_log.csv`
- keep the dashboard aligned with current full-scale evidence
- absorb new raw data as it arrives
- begin writing Introduction and Literature Review now
- document failed methods instead of discarding them mentally
