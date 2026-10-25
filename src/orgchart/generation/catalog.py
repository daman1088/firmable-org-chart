"""Generator-side vocabulary: departments, title templates, archetypes, geography.

This module is private to the generator. Inference builds its own lexicon
independently (Phase 2) and must not import it, otherwise the evaluation would
reward memorising the generator rather than understanding titles.

Title templates use three placeholders:
    {role}  a role noun for the specialty, e.g. "Backend Engineer"
    {spec}  the specialty / team name, e.g. "Payments" (falls back to {func})
    {func}  the department's function name, e.g. "Engineering"
"""

from __future__ import annotations

from dataclasses import dataclass, field

Templates = dict[int, tuple[str, ...]]

# ----------------------------------------------------------------------------- defaults
DEFAULT_IC: Templates = {
    0: ("{role} Intern", "{func} Intern", "Intern, {spec}"),
    1: ("Junior {role}", "Associate {role}", "Graduate {role}"),
    2: ("{role}",),
    3: ("Senior {role}",),
    4: ("Lead {role}", "Principal {role}"),
}

DEFAULT_MGR: Templates = {
    4: ("Team Lead, {spec}", "{spec} Team Lead", "Team Leader - {spec}"),
    5: ("{spec} Manager", "Manager, {spec}"),
    6: ("Senior {spec} Manager", "Senior Manager, {spec}"),
    7: ("Director of {spec}", "Head of {spec}", "Director, {spec}"),
    8: ("Senior Director, {spec}", "Head of {spec}"),
    9: ("VP {spec}", "Vice President, {spec}"),
    10: ("SVP {spec}", "Senior Vice President, {spec}"),
}

DEFAULT_HEAD: Templates = {
    4: ("{func} Lead", "{func} Team Lead"),
    5: ("{func} Manager", "Head of {func}", "{func} Lead"),
    6: ("Senior {func} Manager", "Head of {func}"),
    7: ("Head of {func}", "Director of {func}", "{func} Director"),
    8: ("Head of {func}", "Senior Director of {func}", "General Manager, {func}"),
    9: ("VP {func}", "VP of {func}", "Vice President, {func}", "General Manager, {func}"),
    10: ("SVP {func}", "Senior Vice President, {func}", "EVP {func}", "Executive Vice President, {func}"),
}

DEFAULT_IC_LEVEL_WEIGHTS: dict[int, float] = {0: 0.04, 1: 0.18, 2: 0.38, 3: 0.30, 4: 0.10}
FRONTLINE_IC_LEVEL_WEIGHTS: dict[int, float] = {0: 0.03, 1: 0.37, 2: 0.35, 3: 0.19, 4: 0.06}


@dataclass(frozen=True)
class DeptSpec:
    key: str
    func: str
    labels: tuple[str, ...]  # how the raw `department` field may be written; first is canonical
    specialties: dict[str, tuple[str, ...]]  # specialty -> role nouns
    ic: Templates = field(default_factory=dict)
    mgr: Templates = field(default_factory=dict)
    head: Templates = field(default_factory=dict)
    c_titles: tuple[str, ...] = ()
    c_propensity: float = 0.0  # base chance the head is C-level (scaled by size)
    ic_team_cap: int = 9  # max ICs a frontline manager/lead runs directly
    ic_level_weights: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_IC_LEVEL_WEIGHTS))
    spec_overrides: dict[str, dict[str, Templates]] = field(default_factory=dict)

    def ic_templates(self, level: int, specialty: str | None) -> tuple[str, ...]:
        override = self.spec_overrides.get(specialty or "", {}).get("ic", {})
        return override.get(level) or self.ic.get(level) or DEFAULT_IC[level]

    def mgr_templates(self, level: int, specialty: str | None) -> tuple[str, ...]:
        override = self.spec_overrides.get(specialty or "", {}).get("mgr", {})
        return override.get(level) or self.mgr.get(level) or DEFAULT_MGR[level]

    def head_templates(self, level: int) -> tuple[str, ...]:
        if level >= 11:
            return self.c_titles or DEFAULT_HEAD[10]
        return self.head.get(level) or DEFAULT_HEAD[level]


DEPARTMENTS: dict[str, DeptSpec] = {
    d.key: d
    for d in [
        DeptSpec(
            key="engineering",
            func="Engineering",
            labels=("Engineering", "Technology", "R&D", "Software Engineering", "Product & Engineering", "Tech"),
            specialties={
                "Backend": ("Backend Engineer", "Software Engineer"),
                "Frontend": ("Frontend Engineer", "Software Engineer"),
                "Mobile": ("Mobile Engineer", "iOS Engineer", "Android Engineer"),
                "Platform": ("Platform Engineer", "DevOps Engineer", "Site Reliability Engineer"),
                "Infrastructure": ("Infrastructure Engineer", "Cloud Engineer"),
                "QA": ("QA Engineer", "Test Engineer", "Software Engineer in Test"),
                "Security": ("Security Engineer",),
                "Data Platform": ("Data Engineer",),
            },
            ic={
                2: ("{role}", "{role} II"),
                3: ("Senior {role}", "{role} III"),
                4: ("Staff {role}", "Principal {role}"),
            },
            mgr={
                4: ("Tech Lead, {spec}", "{spec} Tech Lead", "Engineering Team Lead"),
                5: ("Engineering Manager", "Engineering Manager, {spec}", "{spec} Engineering Manager"),
                6: ("Senior Engineering Manager", "Senior Engineering Manager, {spec}"),
                7: ("Director of Engineering", "Director of Engineering, {spec}", "Head of {spec} Engineering"),
                8: ("Senior Director of Engineering", "Head of {spec} Engineering"),
                9: ("VP Engineering, {spec}", "VP of {spec} Engineering"),
            },
            head={
                5: ("Engineering Manager", "Head of Engineering"),
                6: ("Senior Engineering Manager", "Head of Engineering"),
                7: ("Director of Engineering", "Head of Engineering"),
                8: ("Head of Engineering", "Senior Director of Engineering"),
                9: ("VP Engineering", "VP of Engineering", "Vice President of Engineering"),
                10: ("SVP Engineering", "SVP of Engineering"),
            },
            c_titles=("Chief Technology Officer", "CTO"),
            c_propensity=0.5,
        ),
        DeptSpec(
            key="product",
            func="Product",
            labels=("Product", "Product Management", "Product & Engineering"),
            specialties={
                "Core Product": ("Product Manager",),
                "Growth": ("Product Manager", "Growth Product Manager"),
                "Platform": ("Technical Product Manager", "Product Manager"),
                "Product Operations": ("Product Operations Specialist", "Product Analyst"),
            },
            ic={1: ("Associate Product Manager", "Junior {role}"), 4: ("Principal {role}", "Lead {role}")},
            mgr={
                4: ("Product Lead, {spec}",),
                5: ("Group Product Manager", "Product Lead, {spec}"),
                6: ("Senior Group Product Manager", "Head of Product, {spec}"),
                7: ("Director of Product, {spec}", "Director of Product Management"),
                8: ("Senior Director of Product",),
            },
            head={7: ("Head of Product", "Director of Product"), 9: ("VP Product", "VP of Product")},
            c_titles=("Chief Product Officer",),
            c_propensity=0.15,
        ),
        DeptSpec(
            key="design",
            func="Design",
            labels=("Design", "Product Design", "UX", "Creative"),
            specialties={
                "Product Design": ("Product Designer", "UX Designer", "UI Designer"),
                "Research": ("UX Researcher", "User Researcher"),
                "Brand": ("Brand Designer", "Graphic Designer"),
                "Content Design": ("Content Designer", "UX Writer"),
            },
            mgr={4: ("Design Lead, {spec}",), 5: ("Design Manager", "{spec} Manager")},
            head={5: ("Design Manager", "Design Lead"), 7: ("Head of Design", "Director of Design")},
        ),
        DeptSpec(
            key="data",
            func="Data",
            labels=("Data", "Data & Analytics", "Analytics", "Business Intelligence", "Data Science"),
            specialties={
                "Analytics": ("Data Analyst", "Business Intelligence Analyst"),
                "Data Science": ("Data Scientist",),
                "Machine Learning": ("Machine Learning Engineer",),
                "Analytics Engineering": ("Analytics Engineer",),
            },
            mgr={5: ("Analytics Manager", "Data Science Manager", "Manager, {spec}")},
            head={7: ("Head of Data", "Director of Data & Analytics"), 9: ("VP Data", "VP Data & Analytics")},
            c_titles=("Chief Data Officer",),
            c_propensity=0.05,
        ),
        DeptSpec(
            key="sales",
            func="Sales",
            labels=("Sales", "Revenue", "Commercial", "Sales & Partnerships", "Business Development"),
            specialties={
                "New Business": ("Account Executive", "Sales Executive"),
                "Enterprise": ("Enterprise Account Executive", "Enterprise Sales Executive"),
                "Mid-Market": ("Account Executive", "Mid-Market Account Executive"),
                "Sales Development": ("Sales Development Representative", "Business Development Representative"),
                "Account Management": ("Account Manager", "Key Account Manager"),
            },
            ic={
                1: ("Sales Development Representative", "Business Development Representative", "Junior {role}"),
                4: ("Principal {role}", "Strategic {role}"),
            },
            mgr={
                4: ("Sales Team Lead", "Team Lead, {spec}"),
                5: ("Sales Manager", "{spec} Sales Manager", "Sales Manager, {spec}"),
                6: ("Senior Sales Manager", "Senior Sales Manager, {spec}"),
                7: ("Sales Director", "Director of Sales, {spec}", "Regional Sales Director", "Head of {spec}"),
                8: ("Senior Sales Director", "Head of Sales, {spec}"),
                9: ("VP Sales, {spec}", "Vice President of Sales"),
            },
            head={
                5: ("Sales Manager", "Head of Sales"),
                7: ("Head of Sales", "Sales Director", "Director of Sales"),
                9: ("VP Sales", "VP of Sales", "Vice President, Sales"),
                10: ("SVP Sales", "SVP Global Sales"),
            },
            c_titles=("Chief Revenue Officer", "Chief Commercial Officer"),
            c_propensity=0.25,
            spec_overrides={
                "Sales Development": {
                    "mgr": {4: ("SDR Team Lead", "Sales Development Team Lead"), 5: ("Sales Development Manager",)}
                }
            },
        ),
        DeptSpec(
            key="marketing",
            func="Marketing",
            labels=("Marketing", "Growth", "Marketing & Communications", "Brand"),
            specialties={
                "Demand Generation": ("Demand Generation Specialist", "Growth Marketer"),
                "Content": ("Content Marketing Specialist", "Copywriter", "Content Writer"),
                "Product Marketing": ("Product Marketing Manager",),
                "Brand & Communications": ("Communications Specialist", "Brand Specialist", "PR Specialist"),
                "Digital": ("Digital Marketing Specialist", "SEO Specialist", "Performance Marketing Specialist"),
                "Events": ("Events Specialist", "Events Coordinator"),
            },
            ic={1: ("Marketing Coordinator", "Marketing Assistant", "Junior {role}")},
            mgr={5: ("Marketing Manager", "{spec} Manager", "Manager, {spec}")},
            head={
                5: ("Marketing Manager", "Head of Marketing"),
                7: ("Head of Marketing", "Director of Marketing", "Marketing Director"),
            },
            c_titles=("Chief Marketing Officer", "CMO"),
            c_propensity=0.15,
        ),
        DeptSpec(
            key="customer_success",
            func="Customer Success",
            labels=(
                "Customer Success",
                "Customer Support",
                "Customer Experience",
                "Support",
                "Client Services",
                "Customer Service",
            ),
            specialties={
                "Support": ("Customer Support Specialist", "Customer Support Representative", "Support Agent"),
                "Customer Success": ("Customer Success Manager", "Customer Success Specialist"),
                "Onboarding": ("Onboarding Specialist", "Implementation Consultant"),
                "Technical Support": ("Technical Support Engineer", "Technical Support Specialist"),
            },
            ic={1: ("Customer Service Representative", "Customer Support Associate", "Junior {role}")},
            mgr={
                4: ("Support Team Lead", "Team Leader, {spec}", "{spec} Team Lead"),
                5: ("{spec} Manager", "Customer Support Manager", "Manager, {spec}"),
                7: ("Director of Customer Success", "Head of {spec}", "Head of Customer Experience"),
            },
            head={
                7: ("Head of Customer Success", "Director of Customer Success", "Head of Customer Experience"),
                9: ("VP Customer Success", "VP Customer Experience"),
            },
            c_titles=("Chief Customer Officer",),
            c_propensity=0.05,
            ic_team_cap=12,
        ),
        DeptSpec(
            key="finance",
            func="Finance",
            labels=("Finance", "Finance & Accounting", "Accounting", "Finance & Administration"),
            specialties={
                "Accounting": ("Accountant", "Management Accountant"),
                "FP&A": ("Financial Analyst", "FP&A Analyst"),
                "Accounts Payable": ("Accounts Payable Officer", "Accounts Officer"),
                "Payroll": ("Payroll Officer", "Payroll Specialist"),
                "Tax": ("Tax Accountant",),
                "Treasury": ("Treasury Analyst",),
            },
            ic={1: ("Accounts Assistant", "Junior {role}", "Graduate Accountant")},
            mgr={
                5: ("Finance Manager", "{spec} Manager"),
                6: ("Senior Finance Manager", "Financial Controller"),
                7: ("Financial Controller", "Head of {spec}"),
            },
            head={
                5: ("Finance Manager", "Head of Finance"),
                6: ("Financial Controller", "Senior Finance Manager"),
                7: ("Head of Finance", "Financial Controller"),
                8: ("Group Financial Controller", "Head of Finance"),
                9: ("VP Finance", "Finance Director", "General Manager, Finance"),
            },
            c_titles=("Chief Financial Officer", "CFO"),
            c_propensity=0.45,
        ),
        DeptSpec(
            key="people",
            func="People & Culture",
            labels=("Human Resources", "People", "People & Culture", "HR", "Talent"),
            specialties={
                "Talent Acquisition": ("Recruiter", "Talent Acquisition Partner", "Talent Acquisition Specialist"),
                "HR Business Partnering": ("HR Business Partner", "People Partner"),
                "People Operations": ("People Operations Specialist", "HR Advisor"),
                "Learning & Development": ("Learning & Development Specialist",),
                "Total Rewards": ("Reward Specialist", "Remuneration Analyst"),
            },
            ic={1: ("HR Coordinator", "People & Culture Coordinator", "Recruitment Coordinator")},
            mgr={5: ("HR Manager", "{spec} Manager", "Talent Acquisition Manager")},
            head={
                5: ("HR Manager", "People & Culture Manager"),
                7: ("Head of People", "Head of People & Culture", "HR Director"),
                9: ("VP People", "General Manager, People & Culture"),
            },
            c_titles=("Chief People Officer", "Chief Human Resources Officer"),
            c_propensity=0.15,
        ),
        DeptSpec(
            key="operations",
            func="Operations",
            labels=("Operations", "Business Operations", "Corporate Services", "Administration"),
            specialties={
                "Business Operations": ("Operations Analyst", "Business Operations Analyst"),
                "Facilities": ("Facilities Coordinator", "Office Coordinator"),
                "Procurement": ("Procurement Officer", "Procurement Specialist"),
                "PMO": ("Program Manager", "Project Manager"),
                "Administration": ("Administration Officer", "Administrative Assistant", "Receptionist"),
            },
            ic={1: ("Operations Assistant", "Administration Assistant")},
            mgr={5: ("Operations Manager", "{spec} Manager")},
            head={7: ("Head of Operations", "Director of Operations"), 9: ("VP Operations", "General Manager, Operations")},
            c_titles=("Chief Operating Officer", "COO"),
            c_propensity=0.3,
        ),
        DeptSpec(
            key="legal",
            func="Legal",
            labels=("Legal", "Legal & Compliance", "Legal Affairs"),
            specialties={
                "Commercial": ("Legal Counsel", "Commercial Lawyer"),
                "Corporate": ("Corporate Counsel", "Company Secretary"),
                "Privacy": ("Privacy Counsel",),
                "Legal Operations": ("Paralegal", "Legal Assistant"),
            },
            ic={1: ("Paralegal", "Legal Assistant", "Junior Lawyer"), 4: ("Senior {role}", "Lead Counsel")},
            mgr={5: ("Legal Manager", "Senior Legal Counsel")},
            head={
                5: ("Legal Manager", "Head of Legal"),
                6: ("Head of Legal", "Senior Legal Counsel"),
                7: ("Head of Legal", "Deputy General Counsel"),
                8: ("General Counsel", "Head of Legal"),
                9: ("General Counsel", "VP Legal", "General Counsel & Company Secretary"),
                10: ("General Counsel", "Chief Legal Officer"),
            },
            c_titles=("Chief Legal Officer",),
        ),
        DeptSpec(
            key="it",
            func="IT",
            labels=("IT", "Information Technology", "Technology", "IT Services"),
            specialties={
                "Service Desk": ("IT Support Analyst", "Service Desk Analyst", "Help Desk Technician"),
                "Infrastructure": ("Systems Administrator", "Network Engineer", "Systems Engineer"),
                "Business Systems": ("Business Systems Analyst", "Salesforce Administrator"),
                "Cyber Security": ("Cyber Security Analyst",),
            },
            ic={1: ("IT Support Officer", "Junior {role}", "IT Graduate")},
            mgr={4: ("IT Team Lead", "Service Desk Team Lead"), 5: ("IT Manager", "{spec} Manager")},
            head={
                5: ("IT Manager", "Head of IT"),
                7: ("Head of IT", "IT Director", "Head of Technology"),
                9: ("VP IT", "General Manager, Technology"),
            },
            c_titles=("Chief Information Officer", "CIO"),
            c_propensity=0.1,
        ),
        DeptSpec(
            key="risk_compliance",
            func="Risk & Compliance",
            labels=("Risk", "Compliance", "Risk & Compliance", "Governance, Risk & Compliance"),
            specialties={
                "Compliance": ("Compliance Analyst", "Compliance Officer"),
                "Risk": ("Risk Analyst", "Operational Risk Analyst"),
                "Financial Crime": ("AML Analyst", "Financial Crime Analyst", "KYC Analyst"),
                "Internal Audit": ("Internal Auditor",),
            },
            mgr={5: ("Compliance Manager", "Risk Manager", "{spec} Manager")},
            head={7: ("Head of Risk", "Head of Compliance", "Head of Risk & Compliance"), 9: ("VP Risk & Compliance",)},
            c_titles=("Chief Risk Officer",),
            c_propensity=0.3,
        ),
        DeptSpec(
            key="production",
            func="Production",
            labels=("Production", "Manufacturing", "Operations", "Plant Operations"),
            specialties={
                "Assembly": ("Assembly Operator", "Assembler"),
                "Fabrication": ("Fabricator", "Welder", "CNC Machinist"),
                "Packaging": ("Packaging Operator", "Packer"),
                "Maintenance": ("Maintenance Technician", "Electrician", "Fitter"),
            },
            ic={
                0: ("Apprentice {role}",),
                1: ("Production Worker", "Process Worker", "Trainee {role}"),
                3: ("Senior {role}", "Leading Hand"),
            },
            mgr={
                4: ("Shift Supervisor", "Production Supervisor", "{spec} Supervisor", "Team Leader, {spec}"),
                5: ("Production Manager", "{spec} Manager", "Shift Manager"),
                6: ("Senior Production Manager", "Operations Manager"),
                7: ("Plant Manager", "Site Manager", "Head of Production"),
                8: ("Senior Plant Manager", "Head of Manufacturing"),
                9: ("General Manager, Manufacturing", "Director of Manufacturing"),
            },
            head={
                7: ("Plant Manager", "Head of Production"),
                8: ("Head of Manufacturing", "Senior Plant Manager"),
                9: ("VP Manufacturing", "General Manager, Manufacturing"),
                10: ("SVP Operations", "EVP Manufacturing"),
            },
            c_titles=("Chief Operating Officer", "COO"),
            c_propensity=0.3,
            ic_team_cap=14,
            ic_level_weights=dict(FRONTLINE_IC_LEVEL_WEIGHTS),
        ),
        DeptSpec(
            key="supply_chain",
            func="Supply Chain",
            labels=("Supply Chain", "Logistics", "Warehouse", "Distribution"),
            specialties={
                "Logistics": ("Logistics Coordinator", "Logistics Officer"),
                "Warehouse": ("Warehouse Operator", "Forklift Driver", "Storeperson", "Warehouse Associate"),
                "Planning": ("Demand Planner", "Supply Planner"),
                "Purchasing": ("Purchasing Officer", "Buyer"),
            },
            mgr={
                4: ("Warehouse Supervisor", "Team Leader, {spec}"),
                5: ("Warehouse Manager", "Logistics Manager", "{spec} Manager"),
                6: ("Distribution Centre Manager",),
            },
            head={7: ("Head of Supply Chain", "Director of Supply Chain"), 9: ("VP Supply Chain", "General Manager, Supply Chain")},
            c_titles=("Chief Supply Chain Officer",),
            c_propensity=0.05,
            ic_team_cap=14,
            ic_level_weights=dict(FRONTLINE_IC_LEVEL_WEIGHTS),
        ),
        DeptSpec(
            key="quality",
            func="Quality",
            labels=("Quality", "Quality Assurance", "QHSE", "Quality & Safety"),
            specialties={
                "Quality Assurance": ("Quality Inspector", "Quality Assurance Officer"),
                "Quality Engineering": ("Quality Engineer",),
                "Regulatory Affairs": ("Regulatory Affairs Specialist",),
                "Health & Safety": ("WHS Officer", "Health & Safety Advisor"),
            },
            mgr={5: ("Quality Manager", "{spec} Manager", "HSE Manager")},
            head={5: ("Quality Manager",), 7: ("Head of Quality", "Quality Director")},
        ),
        DeptSpec(
            key="mfg_engineering",
            func="Engineering",
            labels=("Engineering", "Technical", "Process Engineering"),
            specialties={
                "Process": ("Process Engineer",),
                "Mechanical": ("Mechanical Engineer", "Design Engineer"),
                "Electrical": ("Electrical Engineer", "Controls Engineer"),
                "Drafting": ("Draftsperson", "CAD Designer"),
            },
            mgr={5: ("Engineering Manager", "{spec} Engineering Manager")},
            head={7: ("Head of Engineering", "Engineering Director"), 9: ("VP Engineering", "General Manager, Engineering")},
            c_titles=("Chief Engineer",),
            c_propensity=0.05,
        ),
        DeptSpec(
            key="clinical",
            func="Clinical Services",
            labels=("Clinical", "Clinical Services", "Nursing", "Patient Care", "Medical"),
            specialties={
                "Nursing": ("Registered Nurse",),
                "Allied Health": ("Physiotherapist", "Occupational Therapist", "Dietitian", "Speech Pathologist"),
                "Pharmacy": ("Pharmacist", "Pharmacy Technician"),
                "Medical Services": ("Medical Officer", "General Practitioner", "Resident Medical Officer"),
                "Diagnostics": ("Radiographer", "Sonographer", "Medical Laboratory Scientist"),
            },
            ic={0: ("Clinical Placement Student", "{role} Student"), 4: ("Senior {role}", "Clinical Educator")},
            mgr={
                4: ("Clinical Team Leader", "Team Leader, {spec}"),
                5: ("{spec} Manager", "Clinical Manager, {spec}"),
                6: ("Senior Clinical Manager", "Department Manager, {spec}"),
                7: ("Director of {spec}", "Head of {spec}"),
                8: ("Director of Clinical Services",),
                9: ("Executive Director, Clinical Services",),
            },
            head={
                7: ("Director of Clinical Services", "Head of Clinical Services"),
                9: ("Executive Director, Clinical Services", "General Manager, Clinical Operations"),
            },
            c_titles=("Chief Medical Officer", "Chief Clinical Officer"),
            c_propensity=0.4,
            ic_team_cap=18,
            ic_level_weights=dict(FRONTLINE_IC_LEVEL_WEIGHTS),
            spec_overrides={
                "Nursing": {
                    "ic": {
                        0: ("Student Nurse",),
                        1: ("Enrolled Nurse", "Graduate Registered Nurse", "Assistant in Nursing"),
                        3: ("Clinical Nurse", "Senior Registered Nurse"),
                        4: ("Clinical Nurse Specialist", "Clinical Nurse Consultant", "Nurse Educator"),
                    },
                    "mgr": {
                        4: ("Associate Nurse Unit Manager", "Clinical Nurse Manager"),
                        5: ("Nurse Unit Manager",),
                        6: ("Nursing Manager", "Senior Nurse Manager"),
                        7: ("Director of Nursing", "Assistant Director of Nursing"),
                    },
                }
            },
        ),
        DeptSpec(
            key="store_operations",
            func="Retail Operations",
            labels=("Retail", "Store Operations", "Stores", "Retail Operations"),
            specialties={"Stores": ("Retail Assistant", "Sales Assistant", "Customer Service Assistant")},
            ic={
                0: ("Casual Sales Assistant", "Christmas Casual"),
                1: ("{role}", "Retail Sales Assistant"),
                2: ("Sales Associate", "Senior Sales Assistant", "Visual Merchandiser"),
                3: ("Key Holder", "Senior Sales Associate"),
                4: ("Visual Merchandising Lead", "Store Specialist"),
            },
            mgr={
                4: ("Assistant Store Manager", "Store Supervisor", "Store 2IC"),
                5: ("Store Manager",),
                6: ("Area Manager", "District Manager", "Multi-Site Manager"),
                7: ("Regional Manager", "Regional Operations Manager", "State Manager"),
                8: ("National Retail Operations Manager", "Head of Stores"),
                9: ("General Manager, Retail", "Director of Retail Operations"),
            },
            head={
                7: ("Head of Retail Operations", "Retail Operations Manager"),
                8: ("Head of Stores", "National Retail Manager"),
                9: ("General Manager, Retail", "Head of Retail Operations"),
                10: ("SVP Retail", "EVP Stores"),
            },
            c_titles=("Chief Operating Officer", "Chief Retail Officer"),
            c_propensity=0.3,
            ic_team_cap=18,
            ic_level_weights=dict(FRONTLINE_IC_LEVEL_WEIGHTS),
        ),
        DeptSpec(
            key="merchandising",
            func="Merchandise",
            labels=("Merchandise", "Merchandising", "Buying", "Product"),
            specialties={
                "Buying": ("Buyer", "Assistant Buyer"),
                "Planning": ("Merchandise Planner", "Allocation Analyst"),
                "Visual Merchandising": ("Visual Merchandiser",),
            },
            mgr={5: ("Buying Manager", "Merchandise Planning Manager", "{spec} Manager")},
            head={7: ("Head of Buying", "Head of Merchandise", "Merchandise Director"), 9: ("General Manager, Merchandise",)},
            c_titles=("Chief Merchandising Officer",),
            c_propensity=0.05,
        ),
        DeptSpec(
            key="consulting",
            func="Consulting",
            labels=("Consulting", "Advisory", "Professional Services", "Client Services"),
            specialties={
                "Strategy": ("Consultant", "Strategy Consultant"),
                "Technology": ("Consultant", "Technology Consultant"),
                "Risk & Regulatory": ("Consultant", "Risk Consultant"),
                "Operations": ("Consultant", "Operations Consultant"),
                "Data & Analytics": ("Consultant", "Data Consultant"),
                "Digital": ("Consultant", "Digital Consultant"),
            },
            ic={
                0: ("Vacationer", "Summer Intern"),
                1: ("Analyst", "Business Analyst", "Graduate Consultant"),
                2: ("{role}",),
                3: ("Senior Consultant", "Senior {role}"),
                4: ("Principal Consultant", "Lead Consultant"),
            },
            mgr={
                4: ("Engagement Lead", "Delivery Lead"),
                5: ("Manager", "Manager, {spec}", "Engagement Manager"),
                6: ("Senior Manager", "Senior Manager, {spec}"),
                7: ("Director", "Director, {spec}", "Associate Director"),
                8: ("Associate Partner", "Principal", "Executive Director"),
                9: ("Partner", "Partner, {spec}", "Practice Lead, {spec}"),
                10: ("Senior Partner", "Lead Partner, {spec}"),
            },
            head={9: ("Partner", "Head of Consulting"), 10: ("Senior Partner", "Head of Consulting")},
            c_titles=("Managing Partner, Consulting", "Head of Consulting"),
            c_propensity=0.1,
            ic_team_cap=8,
        ),
        DeptSpec(
            key="business_development",
            func="Business Development",
            labels=("Business Development", "Growth", "Partnerships", "Sales"),
            specialties={
                "Partnerships": ("Partnerships Manager", "Alliance Manager"),
                "Bids": ("Bid Manager", "Proposal Writer", "Bid Coordinator"),
                "Business Development": ("Business Development Manager", "Business Development Executive"),
            },
            mgr={5: ("Business Development Lead", "Head of {spec}")},
            head={7: ("Head of Business Development", "Director of Partnerships"), 9: ("VP Business Development",)},
            c_titles=("Chief Growth Officer",),
            c_propensity=0.05,
        ),
    ]
}

# Executive office: CEO, executive assistants, chief of staff (not a growable department).
EXECUTIVE_LABELS = ("Executive", "Office of the CEO", "Executive Office", "Leadership", "Management")
BOARD_LABELS = ("Board", "Board of Directors", "Advisory Board")

EA_TITLES_CEO = (
    "Executive Assistant to the CEO",
    "Executive Assistant",
    "EA to the CEO",
    "Executive Assistant to the Managing Director",
    "Personal Assistant to the CEO",
)
EA_TITLES_EXEC = ("Executive Assistant to the {exec}", "Executive Assistant", "EA to the {exec}")
CHIEF_OF_STAFF_TITLES = ("Chief of Staff", "Chief of Staff to the CEO", "Chief of Staff, Office of the CEO")
BOARD_TITLES = (
    "Non-Executive Director",
    "Board Member",
    "Advisor",
    "Board Advisor",
    "Chairman of the Board",
    "Investor & Board Member",
    "Strategic Advisor",
)
VAGUE_TITLES = ("Manager", "Team Member", "Staff Member", "Employee", "Consultant", "Specialist", "Associate", "Leader")

# Which department heads may report into another department's head instead of the CEO.
OWNERSHIP: dict[str, tuple[tuple[str, float], ...]] = {
    "product": (("engineering", 0.2),),
    "design": (("product", 0.7), ("engineering", 0.1)),
    "data": (("engineering", 0.4), ("product", 0.2)),
    "customer_success": (("sales", 0.4), ("operations", 0.2)),
    "marketing": (("sales", 0.25),),
    "legal": (("finance", 0.4),),
    "it": (("finance", 0.3), ("operations", 0.3), ("engineering", 0.2)),
    "people": (("operations", 0.2), ("finance", 0.1)),
    "risk_compliance": (("finance", 0.3), ("legal", 0.3)),
    "supply_chain": (("production", 0.4), ("operations", 0.4)),
    "quality": (("production", 0.5), ("clinical", 0.3)),
    "merchandising": (("store_operations", 0.2),),
    "business_development": (("marketing", 0.3),),
    "mfg_engineering": (("production", 0.5),),
}


# ----------------------------------------------------------------------------- archetypes
@dataclass(frozen=True)
class Archetype:
    key: str
    industry_label: str  # observed `industry` field
    dept_weights: dict[str, float]
    core_depts: tuple[str, ...]
    divisions: tuple[str, ...]
    divisional_depts: tuple[str, ...]
    regional_depts: tuple[str, ...]
    founded_range: tuple[int, int]
    name_words: tuple[str, ...]
    ceo_titles: tuple[tuple[str, float], ...]
    founder_probability: float


ARCHETYPES: dict[str, Archetype] = {
    a.key: a
    for a in [
        Archetype(
            key="saas",
            industry_label="Software Development",
            dept_weights={
                "engineering": 30, "product": 7, "design": 3, "data": 4, "sales": 15, "marketing": 8,
                "customer_success": 12, "finance": 4, "people": 4, "operations": 4, "legal": 1.5, "it": 2,
            },
            core_depts=("engineering", "product", "sales", "marketing", "customer_success", "finance", "people"),
            divisions=("Enterprise", "SMB", "Platform"),
            divisional_depts=("engineering", "product", "sales"),
            regional_depts=("sales", "customer_success"),
            founded_range=(2006, 2021),
            name_words=("Cloud", "Labs", "Software", "Systems", "HQ", "AI", "Stack", "Works"),
            ceo_titles=(("Chief Executive Officer", 0.45), ("CEO", 0.2), ("Co-Founder & CEO", 0.25), ("Founder & CEO", 0.1)),
            founder_probability=0.35,
        ),
        Archetype(
            key="fintech",
            industry_label="Financial Services",
            dept_weights={
                "engineering": 25, "product": 6, "design": 2, "data": 5, "sales": 10, "marketing": 6,
                "customer_success": 12, "risk_compliance": 10, "finance": 6, "people": 4, "operations": 6,
                "legal": 3, "it": 3,
            },
            core_depts=("engineering", "product", "risk_compliance", "customer_success", "finance", "people", "sales"),
            divisions=("Payments", "Lending", "Wealth"),
            divisional_depts=("engineering", "product", "sales", "risk_compliance"),
            regional_depts=("sales", "customer_success"),
            founded_range=(2008, 2021),
            name_words=("Pay", "Capital", "Finance", "Money", "Ledger", "Wealth", "Lend", "Bank"),
            ceo_titles=(("Chief Executive Officer", 0.5), ("CEO", 0.15), ("Co-Founder & CEO", 0.25), ("Managing Director", 0.1)),
            founder_probability=0.3,
        ),
        Archetype(
            key="manufacturing",
            industry_label="Manufacturing",
            dept_weights={
                "production": 38, "supply_chain": 14, "quality": 6, "mfg_engineering": 8, "sales": 8,
                "customer_success": 4, "finance": 5, "people": 4, "it": 3, "legal": 1, "marketing": 2,
            },
            core_depts=("production", "supply_chain", "quality", "sales", "finance", "people"),
            divisions=("Industrial", "Commercial", "Residential"),
            divisional_depts=("production", "sales", "mfg_engineering"),
            regional_depts=("production", "sales", "supply_chain"),
            founded_range=(1950, 2012),
            name_words=("Industries", "Manufacturing", "Engineering", "Metals", "Plastics", "Fabrication", "Group"),
            ceo_titles=(("Chief Executive Officer", 0.45), ("Managing Director", 0.35), ("CEO", 0.1), ("General Manager", 0.1)),
            founder_probability=0.1,
        ),
        Archetype(
            key="healthcare",
            industry_label="Hospitals and Health Care",
            dept_weights={
                "clinical": 50, "operations": 10, "quality": 4, "finance": 6, "people": 6, "it": 5,
                "marketing": 2, "legal": 1.5, "data": 2,
            },
            core_depts=("clinical", "operations", "finance", "people", "it"),
            divisions=("Hospitals", "Primary Care", "Aged Care"),
            divisional_depts=("clinical", "operations"),
            regional_depts=("clinical",),
            founded_range=(1960, 2015),
            name_words=("Health", "Care", "Medical", "Clinics", "Wellness", "Healthcare"),
            ceo_titles=(("Chief Executive Officer", 0.6), ("CEO", 0.2), ("Managing Director", 0.2)),
            founder_probability=0.05,
        ),
        Archetype(
            key="retail",
            industry_label="Retail",
            dept_weights={
                "store_operations": 50, "merchandising": 6, "supply_chain": 14, "marketing": 6, "engineering": 5,
                "customer_success": 6, "finance": 4, "people": 5, "it": 3, "legal": 1,
            },
            core_depts=("store_operations", "merchandising", "supply_chain", "marketing", "finance", "people"),
            divisions=("Apparel", "Home", "Online"),
            divisional_depts=("store_operations", "merchandising"),
            regional_depts=("store_operations", "supply_chain"),
            founded_range=(1970, 2016),
            name_words=("Retail", "Stores", "Co", "Supply", "Living", "Outfitters", "Market"),
            ceo_titles=(("Chief Executive Officer", 0.55), ("CEO", 0.15), ("Managing Director", 0.3)),
            founder_probability=0.1,
        ),
        Archetype(
            key="consulting",
            industry_label="Business Consulting and Services",
            dept_weights={
                "consulting": 70, "business_development": 6, "marketing": 4, "finance": 5, "people": 7,
                "operations": 4, "it": 3, "legal": 1.5,
            },
            core_depts=("consulting", "business_development", "finance", "people"),
            divisions=("Public Sector", "Financial Services", "Energy & Resources"),
            divisional_depts=("consulting",),
            regional_depts=("consulting",),
            founded_range=(1975, 2016),
            name_words=("Partners", "Advisory", "Consulting", "Group", "Associates", "Strategy"),
            ceo_titles=(("Chief Executive Officer", 0.4), ("Managing Partner", 0.4), ("CEO & Managing Partner", 0.2)),
            founder_probability=0.15,
        ),
    ]
}

# ----------------------------------------------------------------------------- geography
@dataclass(frozen=True)
class City:
    name: str
    state: str  # state / region label used for domestic regional structures
    country: str


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    global_region: str
    legal_suffixes: tuple[str, ...]
    tld: str
    name_locales: tuple[tuple[str, float], ...]  # Faker locales used for employee names
    cities: tuple[City, ...]


def _cities(country: str, *pairs: tuple[str, str]) -> tuple[City, ...]:
    return tuple(City(name, state, country) for name, state in pairs)


COUNTRIES: dict[str, Country] = {
    c.code: c
    for c in [
        Country(
            "AU", "Australia", "ANZ", ("Pty Ltd", "Limited", ""), "com.au",
            (("en_AU", 0.6), ("en_GB", 0.1), ("en_IN", 0.1), ("en_US", 0.1), ("en_NZ", 0.1)),
            _cities(
                "AU", ("Sydney", "New South Wales"), ("Melbourne", "Victoria"), ("Brisbane", "Queensland"),
                ("Perth", "Western Australia"), ("Adelaide", "South Australia"), ("Canberra", "Australian Capital Territory"),
                ("Gold Coast", "Queensland"), ("Newcastle", "New South Wales"), ("Hobart", "Tasmania"),
            ),
        ),
        Country(
            "NZ", "New Zealand", "ANZ", ("Limited", "Ltd", ""), "co.nz",
            (("en_NZ", 0.7), ("en_AU", 0.1), ("en_GB", 0.1), ("en_IN", 0.1)),
            _cities("NZ", ("Auckland", "Auckland"), ("Wellington", "Wellington"), ("Christchurch", "Canterbury")),
        ),
        Country(
            "US", "United States", "North America", ("Inc.", "LLC", ""), "com",
            (("en_US", 0.8), ("en_IN", 0.1), ("en_GB", 0.1)),
            _cities(
                "US", ("New York", "East"), ("Boston", "East"), ("San Francisco", "West"), ("Seattle", "West"),
                ("Austin", "Central"), ("Chicago", "Central"),
            ),
        ),
        Country(
            "GB", "United Kingdom", "EMEA", ("Ltd", "Limited", "plc", ""), "co.uk",
            (("en_GB", 0.8), ("en_IN", 0.1), ("en_IE", 0.1)),
            _cities("GB", ("London", "South"), ("Manchester", "North"), ("Birmingham", "Midlands"), ("Edinburgh", "Scotland")),
        ),
        Country(
            "SG", "Singapore", "APAC", ("Pte Ltd", ""), "sg",
            (("en_GB", 0.4), ("en_IN", 0.3), ("en_US", 0.3)),
            _cities("SG", ("Singapore", "Singapore")),
        ),
        Country(
            "IN", "India", "APAC", ("Pvt Ltd", "Limited", ""), "in",
            (("en_IN", 0.9), ("en_GB", 0.1)),
            _cities("IN", ("Bengaluru", "Karnataka"), ("Mumbai", "Maharashtra"), ("Hyderabad", "Telangana"), ("Pune", "Maharashtra")),
        ),
        Country(
            "CA", "Canada", "North America", ("Inc.", "Ltd", ""), "ca",
            (("en_CA", 0.8), ("en_IN", 0.1), ("en_GB", 0.1)),
            _cities("CA", ("Toronto", "Ontario"), ("Vancouver", "British Columbia"), ("Montreal", "Quebec")),
        ),
    ]
}

COMPANY_NAME_PREFIXES = (
    "Apex", "Blue", "Northern", "Southern", "Coastal", "Summit", "Harbour", "Granite", "Silver", "Red", "Green",
    "Iron", "Bright", "Clear", "Pacific", "Atlas", "Nova", "Orbit", "Vertex", "Quantum", "Pioneer", "Evergreen",
    "Horizon", "Kestrel", "Wattle", "Banksia", "Ember", "Lumen", "Cobalt", "Sterling", "Maple", "Cedar", "Falcon",
    "Meridian", "Tidal", "Highland", "Union", "Crown", "Oak", "Willow", "Aurora", "Zenith", "Echo", "Prism",
    "Beacon", "Keystone", "Fusion", "Arrow", "Anchor", "Frontier", "Sapphire", "Copper", "Stone", "River",
)
COMPANY_NAME_MIDDLES = (
    "", "", "", "Point", "Ridge", "Bay", "Peak", "Field", "Line", "Gate", "Bridge", "Wave", "Leaf", "Path", "Rock",
)
