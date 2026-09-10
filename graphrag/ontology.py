"""Controlled vocabulary for the sustainability knowledge graph.

Mirrors sections 3-5 of sustainability_knowledge_graph_construction_prompt.txt.
Used to validate / normalise the LLM's extraction output.
"""
from __future__ import annotations

COMMUNITIES: frozenset[str] = frozenset({
    "Organization", "Reporting", "Environmental", "Social", "Stakeholder",
    "Governance", "Business", "Finance", "Risk_Impact", "Initiative",
    "Compliance_Assurance", "Location", "Measurement",
})

# Which ESG domain each entity type belongs to. This is definitional -- a
# Scope2Emission is Environmental whatever the LLM says -- so it is used to
# OVERRIDE the model's `community` field for known types. Without it the model
# files most numeric entities under "Measurement" (also DEFAULT_COMMUNITY), which
# collapses the domain signal: in one run 671/975 entities were "Measurement" and
# only 7 were "Environmental".
_DOMAIN_GROUPS: dict[str, tuple[str, ...]] = {
    "Organization": (
        "Company", "Subsidiary", "ParentCompany", "JointVenture", "AssociateCompany",
        "BusinessUnit", "Department", "Facility", "Branch", "Project",
    ),
    "Reporting": (
        "Report", "SustainabilityReport", "ESGReport", "BRSRReport", "AnnualReport",
        "ClimateReport", "CSRReport", "ImpactReport", "PolicyDocument", "AssuranceReport",
        "Disclosure", "ReportingPeriod", "FinancialYear", "PublicationDate",
        "ReportingFramework", "Framework", "Standard", "ReportingStandard",
        "FrameworkRequirement", "ReportingRequirement", "Indicator", "Question",
        "ReportingBoundary",
    ),
    "Environmental": (
        "ClimateChange", "ClimateRisk", "ClimateOpportunity", "GHGEmission",
        "Scope1Emission", "Scope2Emission", "Scope3Emission", "EmissionSource",
        "EmissionFactor", "EnergyConsumption", "EnergySource", "RenewableEnergy",
        "WaterWithdrawal", "WaterConsumption", "WaterDischarge", "WaterSource",
        "Waste", "WasteGenerated", "WasteRecycled", "WasteDisposed",
        "Biodiversity", "Ecosystem", "NaturalResource", "LandUse",
    ),
    "Social": (
        "Employee", "Worker", "ContractWorker", "Workforce", "Diversity", "Inclusion",
        "VulnerableGroup", "MarginalizedGroup", "HumanRights", "OccupationalHealth",
        "OccupationalSafety", "Training", "SkillDevelopment", "EmployeeGrievance",
        "LabourRights", "ChildLabour", "ForcedLabour", "Discrimination", "Harassment",
    ),
    "Stakeholder": (
        "Stakeholder", "Customer", "Supplier", "Vendor", "Investor", "Shareholder",
        "Community", "Government", "Regulator", "NGO", "BusinessPartner", "Lender",
        "Borrower",
    ),
    "Governance": (
        "Board", "BoardMember", "Committee", "SustainabilityCommittee", "ESGCommittee",
        "GovernanceStructure", "Policy", "Procedure", "CodeOfConduct", "EthicsPolicy",
        "ESGPolicy", "CSRPolicy", "HumanRightsPolicy", "RiskManagementPolicy",
        "PrivacyPolicy", "AntiCorruptionPolicy", "WhistleblowerPolicy",
    ),
    "Business": (
        "BusinessActivity", "Product", "Service", "CustomerSegment", "Market",
        "Industry", "Sector", "SupplyChain", "ValueChain", "Procurement", "Contract",
        "FinancingActivity",
    ),
    "Finance": (
        "Revenue", "Expense", "Investment", "Asset", "Liability", "FinancialImpact",
        "FinancialRisk", "FinancialOpportunity", "FinancialPerformance", "EconomicValue",
    ),
    "Risk_Impact": (
        "Risk", "Opportunity", "Impact", "MaterialTopic", "MaterialIssue", "ESGRisk",
        "EnvironmentalImpact", "SocialImpact", "GovernanceImpact", "RiskAssessment",
        "MaterialityAssessment", "MaterialityMatrix", "Mitigation", "AdaptationMeasure",
        "CorrectiveAction",
    ),
    "Initiative": (
        "SustainabilityInitiative", "ESGInitiative", "CSRProject", "Program", "Target",
        "ESGTarget", "ClimateTarget", "EmissionReductionTarget", "EnergyTarget",
        "WaterTarget", "WasteTarget", "SocialTarget", "KPI", "Milestone", "Action",
    ),
    "Compliance_Assurance": (
        "Regulation", "Law", "RegulatoryRequirement", "Assurance", "AssuranceProvider",
        "Auditor", "Audit", "Certification", "Certificate", "Assessment", "Verification",
    ),
    "Location": (
        "Country", "State", "District", "City", "Region", "Location", "ProjectSite",
        "OperatingLocation",
    ),
    "Measurement": (
        "Metric", "Measurement", "Value", "Unit", "Methodology", "Calculation",
        "Baseline", "TargetValue", "ActualValue", "Intensity", "DataQuality", "Confidence",
    ),
}

DOMAIN_OF: dict[str, str] = {
    etype: domain for domain, types in _DOMAIN_GROUPS.items() for etype in types
}


# The generic bucket. A type in here says nothing about the ESG domain -- an
# employee headcount and a Scope 2 figure are both "Measurement" -- so for these
# the model's own `community` guess is the better signal and must be kept.
GENERIC_TYPES: frozenset[str] = frozenset(_DOMAIN_GROUPS["Measurement"])


def domain_for(entity_type: str, fallback: str) -> str:
    """Definitional domain for a specific entity type, else the model's own guess.

    Only overrides when the type actually implies a domain: a Scope2Emission is
    Environmental no matter what the model claims. For generic types we defer,
    because overriding there discards real information (it collapsed `Social`
    from 24 entities to 4 in testing).
    """
    if entity_type in GENERIC_TYPES:
        return fallback
    return DOMAIN_OF.get(entity_type, fallback)


# Recover a specific type from the entity name when the model fell back to the
# generic bucket. Ordered most-specific-first; only ever applied to GENERIC_TYPES,
# so this can add precision but never overwrite a considered classification.
_NAME_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("SCOPE 1", "Scope1Emission"),
    ("SCOPE 2", "Scope2Emission"),
    ("SCOPE 3", "Scope3Emission"),
    ("EMISSION INTENSITY", "Intensity"),
    ("GHG EMISSION", "GHGEmission"),
    ("EMISSION", "GHGEmission"),
    ("WATER WITHDRAWAL", "WaterWithdrawal"),
    ("WATER CONSUMPTION", "WaterConsumption"),
    ("WATER DISCHARGE", "WaterDischarge"),
    ("RENEWABLE", "RenewableEnergy"),
    ("ENERGY CONSUMPTION", "EnergyConsumption"),
    ("ENERGY", "EnergyConsumption"),
    ("WASTE RECYCLED", "WasteRecycled"),
    ("WASTE DISPOSED", "WasteDisposed"),
    ("WASTE", "WasteGenerated"),
    ("GRIEVANCE", "EmployeeGrievance"),
    ("TRAINING", "Training"),
    ("PERMANENT EMPLOYEE", "Employee"),
    ("EMPLOYEE", "Employee"),
    ("WORKER", "Worker"),
    ("WORKFORCE", "Workforce"),
    ("TURNOVER", "Revenue"),
    ("REVENUE", "Revenue"),
)


def refine_type(name: str, entity_type: str) -> str:
    """Upgrade a generic 'Measurement' to a specific ontology type from its name."""
    if entity_type not in GENERIC_TYPES:
        return entity_type
    upper = str(name).upper()
    for needle, specific in _NAME_TYPE_RULES:
        if needle in upper:
            return specific
    return entity_type


ENTITY_TYPES: frozenset[str] = frozenset({
    # Organization
    "Company", "Subsidiary", "ParentCompany", "JointVenture", "AssociateCompany",
    "BusinessUnit", "Department", "Facility", "Branch", "Project",
    # Reporting
    "Report", "SustainabilityReport", "ESGReport", "BRSRReport", "AnnualReport",
    "ClimateReport", "CSRReport", "ImpactReport", "PolicyDocument", "AssuranceReport",
    "Disclosure", "ReportingPeriod", "FinancialYear", "PublicationDate",
    "ReportingFramework", "Framework", "Standard", "ReportingStandard",
    "FrameworkRequirement", "ReportingRequirement", "Indicator", "Question",
    "ReportingBoundary",
    # Environmental
    "ClimateChange", "ClimateRisk", "ClimateOpportunity", "GHGEmission",
    "Scope1Emission", "Scope2Emission", "Scope3Emission", "EmissionSource",
    "EmissionFactor", "EnergyConsumption", "EnergySource", "RenewableEnergy",
    "WaterWithdrawal", "WaterConsumption", "WaterDischarge", "WaterSource",
    "Waste", "WasteGenerated", "WasteRecycled", "WasteDisposed",
    "Biodiversity", "Ecosystem", "NaturalResource", "LandUse",
    # Social
    "Employee", "Worker", "ContractWorker", "Workforce", "Diversity", "Inclusion",
    "VulnerableGroup", "MarginalizedGroup", "HumanRights", "OccupationalHealth",
    "OccupationalSafety", "Training", "SkillDevelopment", "EmployeeGrievance",
    "LabourRights", "ChildLabour", "ForcedLabour", "Discrimination", "Harassment",
    # Stakeholder
    "Stakeholder", "Customer", "Supplier", "Vendor", "Investor", "Shareholder",
    "Community", "Government", "Regulator", "NGO", "BusinessPartner", "Lender",
    "Borrower",
    # Governance
    "Board", "BoardMember", "Committee", "SustainabilityCommittee", "ESGCommittee",
    "GovernanceStructure", "Policy", "Procedure", "CodeOfConduct", "EthicsPolicy",
    "ESGPolicy", "CSRPolicy", "HumanRightsPolicy", "RiskManagementPolicy",
    "PrivacyPolicy", "AntiCorruptionPolicy", "WhistleblowerPolicy",
    # Business
    "BusinessActivity", "Product", "Service", "CustomerSegment", "Market",
    "Industry", "Sector", "SupplyChain", "ValueChain", "Procurement", "Contract",
    "FinancingActivity",
    # Finance
    "Revenue", "Expense", "Investment", "Asset", "Liability", "FinancialImpact",
    "FinancialRisk", "FinancialOpportunity", "FinancialPerformance", "EconomicValue",
    # Risk_Impact
    "Risk", "Opportunity", "Impact", "MaterialTopic", "MaterialIssue", "ESGRisk",
    "EnvironmentalImpact", "SocialImpact", "GovernanceImpact", "RiskAssessment",
    "MaterialityAssessment", "MaterialityMatrix", "Mitigation", "AdaptationMeasure",
    "CorrectiveAction",
    # Initiative
    "SustainabilityInitiative", "ESGInitiative", "CSRProject", "Program", "Target",
    "ESGTarget", "ClimateTarget", "EmissionReductionTarget", "EnergyTarget",
    "WaterTarget", "WasteTarget", "SocialTarget", "KPI", "Milestone", "Action",
    # Compliance_Assurance
    "Regulation", "Law", "RegulatoryRequirement", "Assurance", "AssuranceProvider",
    "Auditor", "Audit", "Certification", "Certificate", "Assessment", "Verification",
    # Location
    "Country", "State", "District", "City", "Region", "Location", "ProjectSite",
    "OperatingLocation",
    # Measurement
    "Metric", "Measurement", "Value", "Unit", "Methodology", "Calculation",
    "Baseline", "TargetValue", "ActualValue", "Intensity", "DataQuality", "Confidence",
})

RELATIONSHIPS: frozenset[str] = frozenset({
    "OWNS", "SUBSIDIARY_OF", "PARENT_OF", "ASSOCIATE_OF", "PART_OF", "OPERATES",
    "LOCATED_IN", "HAS_FACILITY", "HAS_BRANCH", "HAS_PROJECT",
    "PUBLISHED", "HAS_REPORT", "REPORTS_ON", "COVERS_PERIOD", "HAS_PUBLICATION_DATE",
    "USES_FRAMEWORK", "ALIGNS_WITH", "REFERENCES_FRAMEWORK", "FOLLOWS_STANDARD",
    "CONTAINS", "HAS_DISCLOSURE", "MAPS_TO", "HAS_REQUIREMENT", "HAS_INDICATOR",
    "ASSURED_BY",
    "EMITS", "GENERATES", "CONSUMES", "USES_ENERGY", "USES_WATER", "WITHDRAWS_WATER",
    "DISCHARGES_WATER", "GENERATES_WASTE", "RECYCLES", "TREATS", "REDUCES",
    "MITIGATES", "PROTECTS", "AFFECTS",
    "EMPLOYS", "TRAINS", "SUPPORTS", "ENGAGES_WITH", "BENEFITS", "RAISES", "REPORTS",
    "PROVIDES", "PARTICIPATES_IN",
    "CONSULTS", "RECEIVES_FEEDBACK_FROM", "RECEIVES_GRIEVANCE_FROM", "SUPPLIES",
    "SERVES", "INVESTS_IN", "FINANCES",
    "GOVERNS", "OVERSEES", "APPROVES", "IMPLEMENTS", "MONITORS", "HAS_POLICY",
    "HAS_PROCEDURE", "APPLIES_TO", "REPORTS_TO",
    "OFFERS", "PERFORMS", "PROCURES_FROM", "OPERATES_IN", "BELONGS_TO",
    "HAS_REVENUE", "HAS_EXPENSE", "HAS_INVESTMENT", "HAS_FINANCIAL_IMPACT",
    "HAS_FINANCIAL_RISK", "HAS_FINANCIAL_OPPORTUNITY", "IMPACTS_FINANCIAL_PERFORMANCE",
    "HAS_RISK", "HAS_OPPORTUNITY", "HAS_IMPACT", "RELATES_TO", "ADAPTS_TO",
    "RESULTS_IN", "ASSESSES", "IDENTIFIES",
    "TARGETS", "HAS_TARGET", "MEASURED_BY", "ACHIEVES", "FUNDS",
    "COMPLIES_WITH", "REQUIRED_BY", "AUDITED_BY", "ASSESSED_BY", "VERIFIED_BY",
    "CERTIFIED_BY", "VIOLATES", "ADDRESSES",
    "HAS_LOCATION", "OCCURS_IN", "AFFECTS_LOCATION",
    "HAS_MEASUREMENT", "HAS_VALUE", "HAS_UNIT", "MEASURES", "HAS_BASELINE",
    "SAME_METRIC_PRIOR_PERIOD",   # derived by merge.py, not extracted
    "HAS_TARGET_VALUE", "CALCULATED_USING", "MEASURED_FOR", "MEASURED_AT",
    "HAS_METHODOLOGY",
})

# Entity types whose instances must NOT be merged across reporting periods.
# Must be a superset of every type `refine_type` can produce -- otherwise
# refining "TOTAL EMPLOYEES" from Measurement to Employee would strip its period
# suffix and silently merge FY24 and FY25 headcounts back together.
PERIOD_SCOPED_TYPES: frozenset[str] = frozenset({
    "Measurement", "Metric", "Value", "ActualValue", "TargetValue", "Baseline", "Intensity",
    "GHGEmission", "Scope1Emission", "Scope2Emission", "Scope3Emission",
    "EnergyConsumption", "RenewableEnergy", "WaterWithdrawal", "WaterConsumption",
    "WaterDischarge", "WasteGenerated", "WasteRecycled", "WasteDisposed",
    "Revenue", "Expense",
    "Employee", "Worker", "Workforce", "Training", "EmployeeGrievance", "Diversity",
})

DEFAULT_ENTITY_TYPE = "Metric"
DEFAULT_COMMUNITY = "Measurement"
DEFAULT_RELATIONSHIP = "RELATES_TO"

# Derived, not extracted. Added by merge.py to chain the same metric across
# reporting periods ("... (FY 2023-24)" -> "... (FY 2024-25)"). Edges carrying it
# are marked derived=True so they are never mistaken for evidence from the text.
SAME_METRIC_PRIOR_PERIOD = "SAME_METRIC_PRIOR_PERIOD"
