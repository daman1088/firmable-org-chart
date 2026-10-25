"""Arrow schemas (data contracts) for every persisted table.

Observed tables are the ONLY inputs inference may read. Truth tables hold the
generator's hidden ground truth and are read exclusively by evaluation code
(see orgchart.storage.truth).
"""

import pyarrow as pa

# --------------------------------------------------------------------------- observed
OBSERVED_COMPANIES = pa.schema(
    [
        pa.field("company_id", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("industry", pa.string(), nullable=False),
        pa.field("hq_city", pa.string(), nullable=False),
        pa.field("hq_country", pa.string(), nullable=False),
        pa.field("headcount_band", pa.string(), nullable=False),  # public "201-500" style band
        pa.field("founded_year", pa.int16(), nullable=False),
        pa.field("offices", pa.list_(pa.string()), nullable=False),
        pa.field("split", pa.string(), nullable=False),  # train | calibration | test
    ]
)

OBSERVED_EMPLOYEES = pa.schema(
    [
        pa.field("employee_id", pa.string(), nullable=False),
        pa.field("company_id", pa.string(), nullable=False),
        pa.field("full_name", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=True),  # raw, noisy job title
        pa.field("department", pa.string(), nullable=True),  # raw, noisy, often missing
        pa.field("location", pa.string(), nullable=True),
        pa.field("start_date", pa.date32(), nullable=True),
    ]
)

# Column-name fragments that must never appear in an observed table.
FORBIDDEN_OBSERVED_FRAGMENTS = ("manager", "report", "level", "depth", "truth", "parent", "is_observed")

# --------------------------------------------------------------------------- truth
TRUTH_COMPANIES = pa.schema(
    [
        pa.field("company_id", pa.string(), nullable=False),
        pa.field("archetype", pa.string(), nullable=False),
        pa.field("scope_kind", pa.string(), nullable=False),  # none | regional | divisional
        pa.field("true_headcount", pa.int32(), nullable=False),
        pa.field("observed_employee_count", pa.int32(), nullable=False),
        pa.field("observed_profile_count", pa.int32(), nullable=False),  # employees + board profiles
        pa.field("coverage_target", pa.float64(), nullable=False),
        pa.field("coverage_actual", pa.float64(), nullable=False),
        pa.field("max_depth", pa.int16(), nullable=False),
        pa.field("departments", pa.list_(pa.string()), nullable=False),
        pa.field("holdout_noise", pa.bool_(), nullable=False),
        pa.field("company_seed", pa.uint64(), nullable=False),
    ]
)

TRUTH_EMPLOYEES = pa.schema(
    [
        pa.field("employee_id", pa.string(), nullable=False),
        pa.field("company_id", pa.string(), nullable=False),
        pa.field("manager_id", pa.string(), nullable=True),  # null for CEO and board profiles
        pa.field("level", pa.int8(), nullable=True),  # null for board profiles
        pa.field("level_name", pa.string(), nullable=True),
        pa.field("department", pa.string(), nullable=False),  # canonical department key
        pa.field("specialty", pa.string(), nullable=True),
        pa.field("scope", pa.string(), nullable=True),  # region / division
        pa.field("role_type", pa.string(), nullable=False),
        pa.field("employment_type", pa.string(), nullable=False),
        pa.field("canonical_title", pa.string(), nullable=False),
        pa.field("is_people_manager", pa.bool_(), nullable=False),
        pa.field("n_direct_reports", pa.int32(), nullable=False),
        pa.field("depth", pa.int16(), nullable=True),  # CEO = 0; null for board profiles
        pa.field("is_observed", pa.bool_(), nullable=False),
        pa.field("nearest_observed_manager_id", pa.string(), nullable=True),
        pa.field("noise_tags", pa.list_(pa.string()), nullable=False),
    ]
)

# --------------------------------------------------------------------------- derived (inference side)
NORMALIZED_TITLES = pa.schema(
    [
        pa.field("employee_id", pa.string(), nullable=False),
        pa.field("company_id", pa.string(), nullable=False),
        pa.field("title_raw", pa.string(), nullable=True),
        pa.field("department_raw", pa.string(), nullable=True),
        pa.field("title_clean", pa.string(), nullable=True),
        pa.field("level", pa.int8(), nullable=True),  # orgchart.ladder.Level
        pa.field("level_name", pa.string(), nullable=True),
        pa.field("level_confidence", pa.float64(), nullable=False),
        pa.field("track", pa.string(), nullable=False),  # executive | manager | ic | support | board | unknown
        pa.field("manages_people", pa.bool_(), nullable=True),
        pa.field("management_confidence", pa.float64(), nullable=False),
        pa.field("department", pa.string(), nullable=True),
        pa.field("department_confidence", pa.float64(), nullable=False),
        pa.field("department_source", pa.string(), nullable=False),
        pa.field("specialty", pa.string(), nullable=True),
        pa.field("specialty_confidence", pa.float64(), nullable=False),
        pa.field("scope", pa.string(), nullable=True),
        pa.field("scope_type", pa.string(), nullable=True),  # region | unit
        pa.field("is_contractor", pa.bool_(), nullable=False),
        pa.field("is_intern", pa.bool_(), nullable=False),
        pa.field("is_interim", pa.bool_(), nullable=False),
        pa.field("is_founder", pa.bool_(), nullable=False),
        pa.field("is_non_employee", pa.bool_(), nullable=False),
        pa.field("parse_confidence", pa.float64(), nullable=False),
        pa.field("reason_codes", pa.list_(pa.string()), nullable=False),
        pa.field("rules_fired", pa.list_(pa.string()), nullable=False),
        pa.field("explanation", pa.list_(pa.string()), nullable=False),
        pa.field("normalizer_version", pa.string(), nullable=False),
    ]
)

ROLE_TYPES = ("ceo", "head", "manager", "ic", "assistant", "chief_of_staff", "board")
EMPLOYMENT_TYPES = ("full_time", "contractor", "intern", "board")
SPLITS = ("train", "calibration", "test")

# ---------------------------------------------------------------------------
# Inference outputs (derived/, Phase 3+). Built from observed data only.
# ---------------------------------------------------------------------------

PREDICTION_STATUSES = ("DIRECT", "ANCESTOR", "AMBIGUOUS", "UNRESOLVED")

PREDICTIONS = pa.schema(
    [
        pa.field("employee_id", pa.string(), nullable=False),
        pa.field("company_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),  # DIRECT | ANCESTOR | AMBIGUOUS | UNRESOLVED
        pa.field("reports_to_id", pa.string(), nullable=True),  # DIRECT only
        pa.field("ancestor_id", pa.string(), nullable=True),  # ANCESTOR only
        pa.field("candidate_ids", pa.list_(pa.string()), nullable=False),  # AMBIGUOUS only
        pa.field("confidence", pa.float64(), nullable=True),  # calibrated probability the status claim is right
        pa.field("top_candidate_id", pa.string(), nullable=True),  # best direct candidate, whatever the status
        pa.field("direct_confidence", pa.float64(), nullable=True),  # best candidate's direct probability
        pa.field("runner_up_id", pa.string(), nullable=True),
        pa.field("runner_up_confidence", pa.float64(), nullable=True),
        pa.field("n_candidates", pa.int32(), nullable=False),
        pa.field("reason_codes", pa.list_(pa.string()), nullable=False),
        pa.field("explanation", pa.list_(pa.string()), nullable=False),
        pa.field("model_variant", pa.string(), nullable=False),
    ]
)
