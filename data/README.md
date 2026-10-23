# Data

The analysis expects the de-identified AWED participant-level analysis file at
`data/df_all_participant.csv`. It is not included in this repository and must
not be committed (see `.gitignore`).

Columns used:

| Column | Meaning |
|---|---|
| `participant_id` | Enrollment-event identifier (used only for deduplication) |
| `cluster` | Residential cluster, 1–24 |
| `intervention` | Randomized arm (1 = Wolbachia deployment, 0 = control) |
| `illness_onset` | Onset date (YYYY-MM-DD); its calendar month is the time index |
| `dengue` | 1 = virologically confirmed dengue, 0 = test-negative |

The file has one row per enrollment event and detected serotype. Seven
identifiers appear twice, differing only in serotype (dual-serotype
infections); the code keeps one row per identifier, giving 6,306 events
(385 test-positive, 5,921 test-negative) over 27 months, and asserts these counts.

Access: the records belong to the AWED Study Group. Requests should follow the
data-sharing statement of the primary AWED report (Utarini et al., *New England
Journal of Medicine* 384:2177–2186, 2021).
