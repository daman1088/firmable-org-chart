"""Title-normalisation lexicon: departments, role nouns, management nouns, abbreviations.

Written independently of the data generator. Everything is lower-case and
matched on whole tokens after cleaning (see clean.py). Industry keys:
saas, fintech, manufacturing, healthcare, retail, consulting, general.
"""

from __future__ import annotations

from dataclasses import dataclass

# ----------------------------------------------------------------------------- industries
INDUSTRY_LABELS: dict[str, str] = {
    "software development": "saas",
    "computer software": "saas",
    "it services and it consulting": "saas",
    "financial services": "fintech",
    "banking": "fintech",
    "manufacturing": "manufacturing",
    "industrial machinery manufacturing": "manufacturing",
    "hospitals and health care": "healthcare",
    "hospital & health care": "healthcare",
    "retail": "retail",
    "business consulting and services": "consulting",
    "management consulting": "consulting",
}

# ----------------------------------------------------------------------------- departments
DEPARTMENTS = (
    "executive", "engineering", "product", "design", "data", "sales", "marketing", "customer_success",
    "finance", "people", "operations", "legal", "it", "risk_compliance", "production", "supply_chain",
    "quality", "hardware_engineering", "clinical", "retail_operations", "merchandising", "consulting",
    "business_development",
)

# Raw `department` field labels -> department. A dict value means "depends on industry".
DEPT_LABELS: dict[str, str | dict[str, str]] = {
    "engineering": {"manufacturing": "hardware_engineering", "*": "engineering"},
    "software engineering": "engineering",
    "r&d": {"manufacturing": "hardware_engineering", "*": "engineering"},
    "research & development": {"manufacturing": "hardware_engineering", "*": "engineering"},
    "tech": "engineering",
    "technology": {"saas": "engineering", "fintech": "engineering", "retail": "engineering", "*": "it"},
    "product & engineering": "engineering",
    "technical": "hardware_engineering",
    "process engineering": "hardware_engineering",
    "it": "it", "information technology": "it", "it services": "it",
    "product": {"retail": "merchandising", "*": "product"},
    "product management": "product",
    "design": "design", "product design": "design", "ux": "design", "creative": "design",
    "data": "data", "data & analytics": "data", "analytics": "data", "business intelligence": "data",
    "data science": "data",
    "sales": {"consulting": "business_development", "*": "sales"},
    "revenue": "sales", "commercial": "sales", "sales & partnerships": "sales",
    "business development": {"consulting": "business_development", "*": "sales"},
    "partnerships": {"consulting": "business_development", "*": "sales"},
    "growth": {"consulting": "business_development", "*": "marketing"},
    "marketing": "marketing", "marketing & communications": "marketing", "brand": "marketing",
    "communications": "marketing",
    "customer success": "customer_success", "customer support": "customer_success",
    "customer experience": "customer_success", "support": "customer_success", "customer service": "customer_success",
    "client services": {"consulting": "consulting", "*": "customer_success"},
    "finance": "finance", "finance & accounting": "finance", "accounting": "finance",
    "finance & administration": "finance",
    "human resources": "people", "people": "people", "people & culture": "people", "hr": "people", "talent": "people",
    "operations": {"manufacturing": "production", "retail": "retail_operations", "*": "operations"},
    "business operations": "operations", "corporate services": "operations", "administration": "operations",
    "legal": "legal", "legal & compliance": "legal", "legal affairs": "legal",
    "risk": "risk_compliance", "compliance": "risk_compliance", "risk & compliance": "risk_compliance",
    "governance, risk & compliance": "risk_compliance",
    "production": "production", "manufacturing": "production", "plant operations": "production",
    "supply chain": "supply_chain", "logistics": "supply_chain", "warehouse": "supply_chain",
    "distribution": "supply_chain",
    "quality": "quality", "quality assurance": "quality", "qhse": "quality", "quality & safety": "quality",
    "clinical": "clinical", "clinical services": "clinical", "nursing": "clinical", "patient care": "clinical",
    "medical": "clinical",
    "retail": "retail_operations", "store operations": "retail_operations", "stores": "retail_operations",
    "retail operations": "retail_operations",
    "merchandise": "merchandising", "merchandising": "merchandising", "buying": "merchandising",
    "consulting": "consulting", "advisory": "consulting", "professional services": "consulting",
    "executive": "executive", "office of the ceo": "executive", "executive office": "executive",
    "leadership": "executive", "management": "executive",
}
BOARD_DEPT_LABELS = {"board", "board of directors", "advisory board"}


def resolve(value: str | dict[str, str], industry: str) -> str:
    if isinstance(value, str):
        return value
    return value.get(industry, value["*"])


# ----------------------------------------------------------------------------- role nouns
@dataclass(frozen=True)
class Role:
    """An individual-contributor role noun phrase."""

    dept: str | dict[str, str] | None
    base_level: int = 2
    specialty: str | None = None
    note: str = ""


R = Role
# Phrase -> Role. Matched longest-first on whole tokens.
IC_ROLES: dict[str, Role] = {
    # engineering
    "software engineer": R("engineering"), "software developer": R("engineering"), "developer": R("engineering"),
    "programmer": R("engineering"), "full stack engineer": R("engineering"),
    "software development engineer": R("engineering"),
    "backend engineer": R("engineering", specialty="Backend"), "back end engineer": R("engineering", specialty="Backend"),
    "backend developer": R("engineering", specialty="Backend"),
    "frontend engineer": R("engineering", specialty="Frontend"), "front end engineer": R("engineering", specialty="Frontend"),
    "frontend developer": R("engineering", specialty="Frontend"),
    "mobile engineer": R("engineering", specialty="Mobile"), "ios engineer": R("engineering", specialty="Mobile"),
    "android engineer": R("engineering", specialty="Mobile"), "ios developer": R("engineering", specialty="Mobile"),
    "android developer": R("engineering", specialty="Mobile"),
    "platform engineer": R("engineering", specialty="Platform"), "devops engineer": R("engineering", specialty="Platform"),
    "site reliability engineer": R("engineering", specialty="Platform"),
    "infrastructure engineer": R("engineering", specialty="Infrastructure"),
    "cloud engineer": R("engineering", specialty="Infrastructure"),
    "quality assurance engineer": R("engineering", specialty="QA"), "test engineer": R("engineering", specialty="QA"),
    "software engineer in test": R("engineering", specialty="QA"), "tester": R("engineering", specialty="QA"),
    "security engineer": R("engineering", specialty="Security"),
    "data engineer": R("engineering", specialty="Data Platform"),
    "engineer": R({"manufacturing": "hardware_engineering", "*": "engineering"}),
    # product / design / data
    "product manager": R("product", note="'Product Manager' is an individual-contributor role despite 'Manager'"),
    "technical product manager": R("product", specialty="Platform", note="IC role despite 'Manager'"),
    "growth product manager": R("product", specialty="Growth", note="IC role despite 'Manager'"),
    "associate product manager": R("product", 1, note="entry-level product IC"),
    "product owner": R("product"), "product analyst": R("product", specialty="Product Operations"),
    "product operations specialist": R("product", specialty="Product Operations"),
    "product designer": R("design", specialty="Product Design"), "ux designer": R("design", specialty="Product Design"),
    "ui designer": R("design", specialty="Product Design"), "designer": R("design"),
    "ux researcher": R("design", specialty="Research"), "user researcher": R("design", specialty="Research"),
    "brand designer": R("design", specialty="Brand"), "graphic designer": R("design", specialty="Brand"),
    "content designer": R("design", specialty="Content Design"), "ux writer": R("design", specialty="Content Design"),
    "data analyst": R("data", specialty="Analytics"), "business intelligence analyst": R("data", specialty="Analytics"),
    "data scientist": R("data", specialty="Data Science"),
    "machine learning engineer": R("data", specialty="Machine Learning"),
    "analytics engineer": R("data", specialty="Analytics Engineering"),
    # sales
    "account executive": R("sales"), "sales executive": R("sales"), "sales representative": R("sales"),
    "sales consultant": R("sales"),
    "enterprise account executive": R("sales", specialty="Enterprise"),
    "enterprise sales executive": R("sales", specialty="Enterprise"),
    "mid market account executive": R("sales", specialty="Mid-Market"),
    "sales development representative": R("sales", 1, "Sales Development", "entry-level sales role"),
    "business development representative": R("sales", 1, "Sales Development", "entry-level sales role"),
    "account manager": R("sales", specialty="Account Management", note="IC role despite 'Manager'"),
    "key account manager": R("sales", specialty="Account Management", note="IC role despite 'Manager'"),
    "relationship manager": R("sales", specialty="Account Management", note="IC role despite 'Manager'"),
    # marketing
    "demand generation specialist": R("marketing", specialty="Demand Generation"),
    "growth marketer": R("marketing", specialty="Demand Generation"),
    "content marketing specialist": R("marketing", specialty="Content"), "copywriter": R("marketing", specialty="Content"),
    "content writer": R("marketing", specialty="Content"),
    "product marketing manager": R("marketing", specialty="Product Marketing", note="IC role despite 'Manager'"),
    "communications specialist": R("marketing", specialty="Brand & Communications"),
    "brand specialist": R("marketing", specialty="Brand & Communications"),
    "pr specialist": R("marketing", specialty="Brand & Communications"),
    "digital marketing specialist": R("marketing", specialty="Digital"), "seo specialist": R("marketing", specialty="Digital"),
    "performance marketing specialist": R("marketing", specialty="Digital"),
    "events specialist": R("marketing", specialty="Events"), "events coordinator": R("marketing", specialty="Events"),
    "marketing coordinator": R("marketing", 1, note="entry-level generalist coordinator"),
    "marketing assistant": R("marketing", 1), "marketing specialist": R("marketing"),
    "marketing executive": R("marketing"),
    # customer success / support
    "customer support specialist": R("customer_success", specialty="Support"),
    "customer support representative": R("customer_success", specialty="Support"),
    "support agent": R("customer_success", specialty="Support"),
    "customer service representative": R({"retail": "retail_operations", "*": "customer_success"}, 1),
    "customer support associate": R("customer_success", 1),
    "customer success manager": R("customer_success", specialty="Customer Success", note="IC role despite 'Manager'"),
    "customer success specialist": R("customer_success", specialty="Customer Success"),
    "onboarding specialist": R("customer_success", specialty="Onboarding"),
    "implementation consultant": R("customer_success", specialty="Onboarding"),
    "technical support engineer": R("customer_success", specialty="Technical Support"),
    "technical support specialist": R("customer_success", specialty="Technical Support"),
    # finance
    "accountant": R("finance", specialty="Accounting"), "management accountant": R("finance", specialty="Accounting"),
    "graduate accountant": R("finance", 1, "Accounting"), "bookkeeper": R("finance"),
    "financial analyst": R("finance", specialty="FP&A"), "fp&a analyst": R("finance", specialty="FP&A"),
    "accounts payable officer": R("finance", specialty="Accounts Payable"),
    "accounts officer": R("finance", specialty="Accounts Payable"), "accounts assistant": R("finance", 1),
    "payroll officer": R("finance", specialty="Payroll"), "payroll specialist": R("finance", specialty="Payroll"),
    "tax accountant": R("finance", specialty="Tax"), "treasury analyst": R("finance", specialty="Treasury"),
    # people
    "recruiter": R("people", specialty="Talent Acquisition"),
    "talent acquisition partner": R("people", specialty="Talent Acquisition", note="'Partner' here is an HR IC role"),
    "talent acquisition specialist": R("people", specialty="Talent Acquisition"),
    "hr business partner": R("people", specialty="HR Business Partnering", note="'Partner' here is an HR IC role"),
    "people partner": R("people", specialty="HR Business Partnering", note="'Partner' here is an HR IC role"),
    "people operations specialist": R("people", specialty="People Operations"),
    "hr advisor": R("people", specialty="People Operations"),
    "learning & development specialist": R("people", specialty="Learning & Development"),
    "reward specialist": R("people", specialty="Total Rewards"), "remuneration analyst": R("people", specialty="Total Rewards"),
    "hr coordinator": R("people", 1), "people & culture coordinator": R("people", 1),
    "recruitment coordinator": R("people", 1), "hr assistant": R("people", 1), "hr generalist": R("people"),
    "hr officer": R("people"),
    # operations
    "operations analyst": R("operations", specialty="Business Operations"),
    "business operations analyst": R("operations", specialty="Business Operations"),
    "facilities coordinator": R("operations", specialty="Facilities"),
    "office coordinator": R("operations", specialty="Facilities"),
    "procurement officer": R("operations", specialty="Procurement"),
    "procurement specialist": R("operations", specialty="Procurement"),
    "program manager": R("operations", specialty="PMO", note="IC role despite 'Manager'"),
    "project manager": R("operations", specialty="PMO", note="IC role despite 'Manager'"),
    "technical program manager": R("engineering", note="IC role despite 'Manager'"),
    "project coordinator": R("operations", specialty="PMO"),
    "administration officer": R("operations", specialty="Administration"),
    "administrative assistant": R("operations", 2, "Administration"),
    "administration assistant": R("operations", 1), "receptionist": R("operations", specialty="Administration"),
    "operations assistant": R({"manufacturing": "production", "*": "operations"}, 1),
    "operations coordinator": R("operations"),
    # legal
    "legal counsel": R("legal", specialty="Commercial"), "commercial lawyer": R("legal", specialty="Commercial"),
    "lawyer": R("legal"), "solicitor": R("legal"), "counsel": R("legal"),
    "corporate counsel": R("legal", specialty="Corporate"), "company secretary": R("legal", specialty="Corporate"),
    "privacy counsel": R("legal", specialty="Privacy"),
    "paralegal": R("legal", 1, "Legal Operations"), "legal assistant": R("legal", 1, "Legal Operations"),
    "junior lawyer": R("legal", 1), "lead counsel": R("legal", 4),
    # it
    "it support analyst": R("it", specialty="Service Desk"), "service desk analyst": R("it", specialty="Service Desk"),
    "help desk technician": R("it", specialty="Service Desk"), "it support officer": R("it", 1),
    "it technician": R("it", specialty="Service Desk"),
    "systems administrator": R("it", specialty="Infrastructure"), "network engineer": R("it", specialty="Infrastructure"),
    "systems engineer": R("it", specialty="Infrastructure"),
    "business systems analyst": R("it", specialty="Business Systems"),
    "salesforce administrator": R("it", specialty="Business Systems"),
    "cyber security analyst": R("it", specialty="Cyber Security"), "it graduate": R("it", 1),
    # risk & compliance
    "compliance analyst": R("risk_compliance", specialty="Compliance"),
    "compliance officer": R("risk_compliance", specialty="Compliance"),
    "risk analyst": R("risk_compliance", specialty="Risk"), "operational risk analyst": R("risk_compliance", specialty="Risk"),
    "aml analyst": R("risk_compliance", specialty="Financial Crime"),
    "financial crime analyst": R("risk_compliance", specialty="Financial Crime"),
    "kyc analyst": R("risk_compliance", specialty="Financial Crime"),
    "internal auditor": R("risk_compliance", specialty="Internal Audit"),
    # production
    "assembly operator": R("production", specialty="Assembly"), "assembler": R("production", specialty="Assembly"),
    "fabricator": R("production", specialty="Fabrication"), "welder": R("production", specialty="Fabrication"),
    "cnc machinist": R("production", specialty="Fabrication"), "machinist": R("production", specialty="Fabrication"),
    "packaging operator": R("production", specialty="Packaging"), "packer": R("production", specialty="Packaging"),
    "maintenance technician": R("production", specialty="Maintenance"),
    "electrician": R("production", specialty="Maintenance"), "fitter": R("production", specialty="Maintenance"),
    "production worker": R("production", 1), "process worker": R("production", 1),
    "machine operator": R("production"), "production operator": R("production"), "operator": R("production"),
    "leading hand": R("production", 3, note="senior hands-on worker; leads a crew without formal reports"),
    # supply chain
    "logistics coordinator": R("supply_chain", specialty="Logistics"),
    "logistics officer": R("supply_chain", specialty="Logistics"),
    "warehouse operator": R("supply_chain", specialty="Warehouse"), "forklift driver": R("supply_chain", specialty="Warehouse"),
    "storeperson": R("supply_chain", specialty="Warehouse"), "warehouse associate": R("supply_chain", specialty="Warehouse"),
    "demand planner": R("supply_chain", specialty="Planning"), "supply planner": R("supply_chain", specialty="Planning"),
    "purchasing officer": R("supply_chain", specialty="Purchasing"),
    "buyer": R({"retail": "merchandising", "*": "supply_chain"}),
    # quality
    "quality inspector": R("quality", specialty="Quality Assurance"),
    "quality assurance officer": R("quality", specialty="Quality Assurance"),
    "quality engineer": R("quality", specialty="Quality Engineering"),
    "regulatory affairs specialist": R("quality", specialty="Regulatory Affairs"),
    "whs officer": R("quality", specialty="Health & Safety"), "health & safety advisor": R("quality", specialty="Health & Safety"),
    "safety officer": R("quality", specialty="Health & Safety"),
    # hardware / manufacturing engineering
    "process engineer": R("hardware_engineering", specialty="Process"),
    "mechanical engineer": R("hardware_engineering", specialty="Mechanical"),
    "design engineer": R("hardware_engineering", specialty="Mechanical"),
    "electrical engineer": R("hardware_engineering", specialty="Electrical"),
    "controls engineer": R("hardware_engineering", specialty="Electrical"),
    "draftsperson": R("hardware_engineering", specialty="Drafting"), "drafter": R("hardware_engineering", specialty="Drafting"),
    "cad designer": R("hardware_engineering", specialty="Drafting"),
    # clinical
    "registered nurse": R("clinical", specialty="Nursing"), "nurse": R("clinical", specialty="Nursing"),
    "enrolled nurse": R("clinical", 1, "Nursing", "enrolled nurses sit below registered nurses"),
    "assistant in nursing": R("clinical", 1, "Nursing"),
    "graduate registered nurse": R("clinical", 1, "Nursing"),
    "clinical nurse": R("clinical", 3, "Nursing", "experienced RN grade"),
    "clinical nurse specialist": R("clinical", 4, "Nursing", "advanced clinical IC, not a manager"),
    "clinical nurse consultant": R("clinical", 4, "Nursing", "advanced clinical IC, not a manager"),
    "nurse educator": R("clinical", 4, "Nursing"), "clinical educator": R("clinical", 4),
    "student nurse": R("clinical", 0, "Nursing"),
    "physiotherapist": R("clinical", specialty="Allied Health"),
    "occupational therapist": R("clinical", specialty="Allied Health"),
    "dietitian": R("clinical", specialty="Allied Health"), "speech pathologist": R("clinical", specialty="Allied Health"),
    "pharmacist": R("clinical", specialty="Pharmacy"), "pharmacy technician": R("clinical", specialty="Pharmacy"),
    "medical officer": R("clinical", specialty="Medical Services"),
    "general practitioner": R("clinical", specialty="Medical Services"),
    "resident medical officer": R("clinical", specialty="Medical Services"), "doctor": R("clinical", specialty="Medical Services"),
    "radiographer": R("clinical", specialty="Diagnostics"), "sonographer": R("clinical", specialty="Diagnostics"),
    "medical laboratory scientist": R("clinical", specialty="Diagnostics"),
    "clinical placement student": R("clinical", 0),
    # retail
    "retail assistant": R("retail_operations", 1, "Stores"),
    "retail sales assistant": R("retail_operations", 1, "Stores"),
    "sales assistant": R({"retail": "retail_operations", "*": "sales"}, 1),
    "customer service assistant": R("retail_operations", 1, "Stores"),
    "sales associate": R({"retail": "retail_operations", "*": "sales"}, 2),
    "visual merchandiser": R("retail_operations", 2),
    "key holder": R("retail_operations", 3, "Stores", "trusted senior store staff; no formal reports"),
    "store specialist": R("retail_operations", 4, "Stores"),
    "visual merchandising lead": R("retail_operations", 4, note="in-store VM lead: senior IC, no formal reports"),
    "christmas casual": R("retail_operations", 0, "Stores"),
    "assistant buyer": R("merchandising", 1, "Buying"),
    "merchandise planner": R("merchandising", specialty="Planning"),
    "allocation analyst": R("merchandising", specialty="Planning"), "allocator": R("merchandising", specialty="Planning"),
    # consulting
    "consultant": R({"consulting": "consulting", "*": None}),
    "strategy consultant": R("consulting", specialty="Strategy"),
    "technology consultant": R("consulting", specialty="Technology"),
    "risk consultant": R("consulting", specialty="Risk & Regulatory"),
    "operations consultant": R("consulting", specialty="Operations"),
    "data consultant": R("consulting", specialty="Data & Analytics"),
    "digital consultant": R("consulting", specialty="Digital"),
    "graduate consultant": R("consulting", 1),
    "vacationer": R("consulting", 0),
    # business development
    "partnerships manager": R({"consulting": "business_development", "*": "sales"}, specialty="Partnerships",
                              note="IC role despite 'Manager'"),
    "alliance manager": R({"consulting": "business_development", "*": "sales"}, specialty="Partnerships",
                          note="IC role despite 'Manager'"),
    "bid manager": R("business_development", specialty="Bids", note="IC role despite 'Manager'"),
    "proposal writer": R("business_development", specialty="Bids"),
    "bid coordinator": R("business_development", specialty="Bids"),
    "business development manager": R({"consulting": "business_development", "*": "sales"},
                                      specialty="Business Development", note="IC role despite 'Manager'"),
    "business development executive": R({"consulting": "business_development", "*": "sales"},
                                        specialty="Business Development"),
    # generic nouns (department must come from elsewhere)
    "analyst": R({"consulting": "consulting", "*": None}),
    "business analyst": R({"consulting": "consulting", "*": "operations"}),
    "specialist": R(None), "coordinator": R(None), "officer": R(None), "associate": R(None),
    "assistant": R(None, 1), "advisor": R(None), "administrator": R(None), "technician": R(None),
    "executive": R(None), "representative": R(None), "scientist": R(None), "partner": R(None),
}

# Roles whose department is a weak hint: a present department field wins over them.
WEAK_ROLE_DEPARTMENTS = {
    "visual merchandiser", "buyer", "engineer", "business analyst", "project manager", "program manager",
    "technical program manager", "operator", "designer", "sales assistant", "sales associate",
    "customer service representative", "operations assistant", "data engineer", "machine learning engineer",
    "systems engineer", "infrastructure engineer", "security engineer", "planner", "consultant",
}

# Consulting grade titles where the industry changes the level (the title alone is generic).
INDUSTRY_ROLE_LEVELS: dict[str, dict[str, int]] = {
    "consulting": {"analyst": 1, "business analyst": 1},
}

# ----------------------------------------------------------------------------- C-suite
C_SUITE: dict[str, tuple[str | dict[str, str], ...]] = {
    # acronym -> possible departments (first = default); dicts resolve by industry
    "ceo": ("executive",),
    "cto": ("engineering",),
    "cfo": ("finance",),
    "coo": ({"manufacturing": "production", "retail": "retail_operations", "*": "operations"},),
    "cmo": ({"healthcare": "clinical", "*": "marketing"}, "marketing"),
    "cro": ({"fintech": "risk_compliance", "*": "sales"}, "sales"),
    "cpo": ("product", "people"),
    "cio": ("it",),
    "chro": ("people",),
    "cdo": ("data",),
    "cco": ("sales", "customer_success"),
    "clo": ("legal",),
    "ciso": ("it",),
}
AMBIGUOUS_ACRONYMS = {"cro", "cpo", "cco", "cmo"}

# "Chief <words> Officer" -> department (checked after removing 'chief' and 'officer')
CHIEF_WORDS: dict[str, str | dict[str, str]] = {
    "executive": "executive",
    "technology": "engineering", "technical": "engineering",
    "financial": "finance", "finance": "finance",
    "operating": {"manufacturing": "production", "retail": "retail_operations", "*": "operations"},
    "marketing": "marketing", "revenue": "sales", "commercial": "sales", "growth": "business_development",
    "product": "product", "people": "people", "human resources": "people", "information": "it",
    "information security": "it", "data": "data", "customer": "customer_success", "legal": "legal",
    "risk": "risk_compliance", "compliance": "risk_compliance", "medical": "clinical", "clinical": "clinical",
    "nursing": "clinical", "supply chain": "supply_chain", "merchandising": "merchandising",
    "retail": "retail_operations", "manufacturing": "production", "engineer": "hardware_engineering",
}

# ----------------------------------------------------------------------------- department keywords
# Words in a title that point at a department (used for manager titles and generic roles).
DEPT_KEYWORDS: dict[str, str | dict[str, str]] = {
    "engineering": {"manufacturing": "hardware_engineering", "*": "engineering"},
    "software": "engineering", "backend": "engineering", "frontend": "engineering", "mobile": "engineering",
    "platform engineering": "engineering", "devops": "engineering", "site reliability": "engineering",
    "quality assurance": {"saas": "engineering", "fintech": "engineering", "*": "quality"},
    "data platform": "engineering", "security engineering": "engineering",
    "product": {"retail": "merchandising", "*": "product"}, "product management": "product",
    "product operations": "product",
    "design": "design", "ux": "design", "research": "design", "brand design": "design",
    "data": "data", "analytics": "data",
    "data science": "data", "machine learning": "data", "business intelligence": "data",
    "analytics engineering": "data",
    "sales": "sales", "sales development": "sales", "account management": "sales", "new business": "sales",
    "enterprise": "sales", "mid market": "sales", "revenue": "sales", "commercial": "sales",
    "marketing": "marketing", "demand generation": "marketing", "content": "marketing",
    "product marketing": "marketing", "brand": "marketing", "communications": "marketing",
    "brand & communications": "marketing",
    "events": "marketing", "growth": "marketing", "digital": "marketing",
    "customer success": "customer_success", "customer support": "customer_success",
    "customer experience": "customer_success", "customer service": "customer_success", "support": "customer_success",
    "onboarding": "customer_success", "technical support": "customer_success",
    "finance": "finance", "financial": "finance", "accounting": "finance", "fp&a": "finance",
    "accounts payable": "finance", "payroll": "finance", "tax": "finance", "treasury": "finance",
    "financial controller": "finance",
    "human resources": "people", "hr": "people", "people": "people", "people & culture": "people",
    "talent acquisition": "people", "talent": "people", "recruitment": "people", "hr business partnering": "people",
    "people operations": "people", "learning & development": "people", "total rewards": "people",
    "operations": {"manufacturing": "production", "retail": "retail_operations", "*": "operations"},
    "business operations": "operations", "facilities": "operations", "procurement": "operations",
    "pmo": "operations", "administration": "operations",
    "legal": "legal", "general counsel": "legal", "privacy": "legal", "legal operations": "legal",
    "corporate": "legal",
    "it": "it", "information technology": "it", "service desk": "it", "infrastructure": "it",
    "business systems": "it", "cyber security": "it", "technology": "it",
    "risk": "risk_compliance", "compliance": "risk_compliance",
    "risk & compliance": "risk_compliance", "financial crime": "risk_compliance", "internal audit": "risk_compliance",
    "risk & regulatory": "consulting",
    "production": "production", "manufacturing": "production", "plant": "production", "shift": "production",
    "assembly": "production", "fabrication": "production", "packaging": "production", "maintenance": "production",
    "supply chain": "supply_chain", "logistics": "supply_chain", "warehouse": "supply_chain",
    "distribution centre": "supply_chain", "distribution center": "supply_chain", "purchasing": "supply_chain",
    "planning": {"retail": "merchandising", "*": "supply_chain"},
    "quality": "quality", "quality engineering": "quality", "hse": "quality", "health & safety": "quality",
    "regulatory affairs": "quality", "whs": "quality",
    "process": "hardware_engineering", "mechanical": "hardware_engineering", "electrical": "hardware_engineering",
    "drafting": "hardware_engineering",
    "clinical": "clinical", "nursing": "clinical", "nurse": "clinical", "allied health": "clinical",
    "pharmacy": "clinical", "medical": "clinical", "medical services": "clinical", "diagnostics": "clinical",
    "clinical services": "clinical", "clinical operations": "clinical",
    "store": "retail_operations", "stores": "retail_operations", "retail": "retail_operations",
    "retail operations": "retail_operations", "area": "retail_operations", "district": "retail_operations",
    "multi site": "retail_operations",
    "merchandise": "merchandising", "merchandising": "merchandising", "buying": "merchandising",
    "visual merchandising": "retail_operations", "merchandise planning": "merchandising",
    "consulting": "consulting", "engagement": "consulting", "delivery": "consulting", "strategy": "consulting",
    "practice": "consulting",
    "business development": {"consulting": "business_development", "*": "sales"},
    "partnerships": {"consulting": "business_development", "*": "sales"},
    "bids": "business_development",
}

# Specialty vocabulary: phrase -> (department, canonical specialty).
SPECIALTIES: dict[str, tuple[str, str]] = {
    "backend": ("engineering", "Backend"), "frontend": ("engineering", "Frontend"),
    "mobile": ("engineering", "Mobile"), "ios": ("engineering", "Mobile"), "android": ("engineering", "Mobile"),
    "platform": ("engineering", "Platform"), "devops": ("engineering", "Platform"),
    "infrastructure": ("engineering", "Infrastructure"), "quality assurance": ("engineering", "QA"),
    "security": ("engineering", "Security"), "data platform": ("engineering", "Data Platform"),
    "core product": ("product", "Core Product"), "growth": ("product", "Growth"),
    "product operations": ("product", "Product Operations"),
    "product design": ("design", "Product Design"), "research": ("design", "Research"),
    "brand": ("design", "Brand"), "content design": ("design", "Content Design"),
    "analytics": ("data", "Analytics"), "data science": ("data", "Data Science"),
    "machine learning": ("data", "Machine Learning"), "analytics engineering": ("data", "Analytics Engineering"),
    "new business": ("sales", "New Business"), "enterprise": ("sales", "Enterprise"),
    "mid market": ("sales", "Mid-Market"), "sales development": ("sales", "Sales Development"),
    "account management": ("sales", "Account Management"),
    "demand generation": ("marketing", "Demand Generation"), "content": ("marketing", "Content"),
    "product marketing": ("marketing", "Product Marketing"),
    "brand & communications": ("marketing", "Brand & Communications"), "digital": ("marketing", "Digital"),
    "events": ("marketing", "Events"),
    "support": ("customer_success", "Support"), "customer success": ("customer_success", "Customer Success"),
    "onboarding": ("customer_success", "Onboarding"), "technical support": ("customer_success", "Technical Support"),
    "accounting": ("finance", "Accounting"), "fp&a": ("finance", "FP&A"),
    "accounts payable": ("finance", "Accounts Payable"), "payroll": ("finance", "Payroll"),
    "tax": ("finance", "Tax"), "treasury": ("finance", "Treasury"),
    "talent acquisition": ("people", "Talent Acquisition"),
    "hr business partnering": ("people", "HR Business Partnering"),
    "people operations": ("people", "People Operations"),
    "learning & development": ("people", "Learning & Development"), "total rewards": ("people", "Total Rewards"),
    "business operations": ("operations", "Business Operations"), "facilities": ("operations", "Facilities"),
    "procurement": ("operations", "Procurement"), "pmo": ("operations", "PMO"),
    "administration": ("operations", "Administration"),
    "commercial": ("legal", "Commercial"), "corporate": ("legal", "Corporate"), "privacy": ("legal", "Privacy"),
    "legal operations": ("legal", "Legal Operations"),
    "service desk": ("it", "Service Desk"), "business systems": ("it", "Business Systems"),
    "cyber security": ("it", "Cyber Security"),
    "compliance": ("risk_compliance", "Compliance"), "risk": ("risk_compliance", "Risk"),
    "financial crime": ("risk_compliance", "Financial Crime"), "internal audit": ("risk_compliance", "Internal Audit"),
    "assembly": ("production", "Assembly"), "fabrication": ("production", "Fabrication"),
    "packaging": ("production", "Packaging"), "maintenance": ("production", "Maintenance"),
    "logistics": ("supply_chain", "Logistics"), "warehouse": ("supply_chain", "Warehouse"),
    "planning": ("supply_chain", "Planning"), "purchasing": ("supply_chain", "Purchasing"),
    "quality engineering": ("quality", "Quality Engineering"),
    "regulatory affairs": ("quality", "Regulatory Affairs"), "health & safety": ("quality", "Health & Safety"),
    "process": ("hardware_engineering", "Process"), "mechanical": ("hardware_engineering", "Mechanical"),
    "electrical": ("hardware_engineering", "Electrical"), "drafting": ("hardware_engineering", "Drafting"),
    "nursing": ("clinical", "Nursing"), "allied health": ("clinical", "Allied Health"),
    "pharmacy": ("clinical", "Pharmacy"), "medical services": ("clinical", "Medical Services"),
    "diagnostics": ("clinical", "Diagnostics"),
    "buying": ("merchandising", "Buying"), "visual merchandising": ("merchandising", "Visual Merchandising"),
    "strategy": ("consulting", "Strategy"), "technology": ("consulting", "Technology"),
    "risk & regulatory": ("consulting", "Risk & Regulatory"), "operations": ("consulting", "Operations"),
    "data & analytics": ("consulting", "Data & Analytics"),
    "partnerships": ("business_development", "Partnerships"), "bids": ("business_development", "Bids"),
    "business development": ("business_development", "Business Development"),
}
# Specialties shared across departments: look these up relative to the resolved department.
SPECIALTY_BY_DEPT_OVERRIDES: dict[tuple[str, str], str] = {
    ("consulting", "digital"): "Digital",
    ("merchandising", "planning"): "Planning",
    ("product", "platform"): "Platform",
    ("quality", "quality assurance"): "Quality Assurance",
    ("it", "infrastructure"): "Infrastructure",
    ("sales", "partnerships"): "Partnerships",
    ("sales", "business development"): "Business Development",
}

# ----------------------------------------------------------------------------- geography (scopes)
REGIONS: dict[str, str] = {
    "apac": "APAC", "asia pacific": "APAC", "emea": "EMEA", "anz": "ANZ", "americas": "Americas",
    "north america": "North America", "latam": "LATAM", "europe": "Europe", "asia": "Asia", "uk": "UK",
    "australia": "Australia", "new zealand": "New Zealand", "global": "Global",
    "new south wales": "New South Wales", "nsw": "New South Wales", "victoria": "Victoria", "vic": "Victoria",
    "queensland": "Queensland", "qld": "Queensland", "western australia": "Western Australia",
    "south australia": "South Australia", "tasmania": "Tasmania", "australian capital territory": "Australian Capital Territory",
    "auckland": "Auckland", "wellington": "Wellington", "canterbury": "Canterbury",
    "karnataka": "Karnataka", "maharashtra": "Maharashtra", "telangana": "Telangana",
    "ontario": "Ontario", "british columbia": "British Columbia", "quebec": "Quebec", "singapore": "Singapore",
    "east": "East", "west": "West", "central": "Central", "north": "North", "south": "South",
    "midlands": "Midlands", "scotland": "Scotland",
}
# Short / generic region words only count as scope when they appear in a qualifier.
QUALIFIER_ONLY_REGIONS = {"east", "west", "central", "north", "south", "vic", "global"}

# ----------------------------------------------------------------------------- abbreviations
TOKEN_EXPANSIONS: dict[str, str] = {
    "sr": "senior", "snr": "senior", "jr": "junior", "jnr": "junior", "mgr": "manager", "mngr": "manager",
    "mgmt": "management", "dir": "director", "engr": "engineer", "asst": "assistant", "assoc": "associate",
    "coord": "coordinator", "princ": "principal", "rep": "representative", "reps": "representative",
    "ops": "operations", "exec": "executive", "acct": "accountant", "mktg": "marketing", "svcs": "services",
    "vp": "vice president", "svp": "senior vice president", "evp": "executive vice president",
    "avp": "assistant vice president", "gm": "general manager", "md": "managing director",
    "swe": "software engineer", "sde": "software development engineer", "sre": "site reliability engineer",
    "qa": "quality assurance", "ae": "account executive", "sdr": "sales development representative",
    "bdr": "business development representative", "bdm": "business development manager",
    "csm": "customer success manager", "ea": "executive assistant", "pa": "personal assistant",
    "tpm": "technical program manager", "pmm": "product marketing manager", "apm": "associate product manager",
    "rn": "registered nurse", "en": "enrolled nurse", "ain": "assistant in nursing", "num": "nurse unit manager",
    "anum": "associate nurse unit manager", "cns": "clinical nurse specialist", "rmo": "resident medical officer",
    "gp": "general practitioner", "ot": "occupational therapist", "hrbp": "hr business partner",
    "ml": "machine learning", "bi": "business intelligence", "ap": "accounts payable",
    "l&d": "learning & development", "p&c": "people & culture", "cx": "customer experience",
    "ux/ui": "ux", "ui/ux": "ux",
}
# Tokens whose expansion depends on position or industry.
CONTEXT_TOKENS = {"eng", "dev", "admin", "pm", "cnc", "hr", "it"}

ROMAN_LEVELS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}

# ----------------------------------------------------------------------------- special whole titles
BOARD_PATTERNS = (
    r"^non executive director$", r"^board member$", r"^member of the board$", r"^(strategic |board )?advis[eo]r$",
    r"^chair(man|woman|person)?( of the board)?$", r"^(investor|investor & board member)$", r"^board director$",
    r"^board of directors$",
)
VAGUE_TITLES = {"manager", "team member", "staff member", "employee", "staff", "worker", "specialist", "associate",
                "professional", "leader", "consultant", "team"}

CHIEF_OF_STAFF = r"\bchief of staff\b"
EXEC_ASSISTANT = r"\b(executive assistant|personal assistant)\b"
