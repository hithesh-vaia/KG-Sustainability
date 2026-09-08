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
    "HAS_TARGET_VALUE", "CALCULATED_USING", "MEASURED_FOR", "MEASURED_AT",
    "HAS_METHODOLOGY",
})

# entity types whose instances must NOT be merged across reporting periods
PERIOD_SCOPED_TYPES: frozenset[str] = frozenset({
    "Measurement", "Value", "ActualValue", "TargetValue", "Baseline", "Intensity",
    "GHGEmission", "Scope1Emission", "Scope2Emission", "Scope3Emission",
    "EnergyConsumption", "WaterWithdrawal", "WaterConsumption", "WaterDischarge",
    "WasteGenerated", "WasteRecycled", "WasteDisposed", "Revenue", "Expense",
})

DEFAULT_ENTITY_TYPE = "Metric"
DEFAULT_COMMUNITY = "Measurement"
DEFAULT_RELATIONSHIP = "RELATES_TO"
