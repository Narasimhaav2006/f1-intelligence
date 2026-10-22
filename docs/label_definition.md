# Podium label definition

Prediction problem: will a driver finish in the top three, using only
information available before the race.

## Label
- `podium = 1` if the final official `position` is 1, 2 or 3. Otherwise 0.
- Disqualified drivers are listed at the back of the classification in the
  source data, so they receive label 0.

## Evaluation population
- Included: every driver who started, including retirements (a retirement is
  not known before the race).
- Excluded: drivers with `positionText = W` (Did not start or Withdrew).
  This is about 1% of rows and none of them could finish on the podium.

## Notes
- `grid = 0` is a real entry (probably a pit-lane start), not a missing value.
  The encoding is decided in the feature engineering step.
- Source: Jolpica-F1 (CC BY-NC-SA 4.0). Data may contain errors.