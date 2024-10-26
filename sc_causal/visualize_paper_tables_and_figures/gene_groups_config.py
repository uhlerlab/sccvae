GENE_MODULES = {
    "ribosomal_proteins_60S": [
        "RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL8", "RPL9", "RPL10",
        "RPL10A", "RPL11", "RPL12", "RPL13", "RPL14", "RPL18", "RPL19",
        "RPL21", "RPL23", "RPL24", "RPL26", "RPL27A", "RPL30", "RPL31",
        "RPL32", "RPL34", "RPL36", "RPL37", "RPL37A", "RPLP2"
    ],
    "ribosomal_proteins_40S": [
        "RPS3", "RPS7", "RPS9", "RPS11", "RPS12", "RPS15A", "RPS17",
        "RPS19", "RPS20", "RPS24", "RPS28", "RPS29", "RPSA"
    ],
    "ribosomal_proteins": [
        "RPS3", "RPS7", "RPS9", "RPS11", "RPS12", "RPS15A", "RPS17",
        "RPS19", "RPS20", "RPS24", "RPS28", "RPS29", "RPSA",
        "RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL8", "RPL9", "RPL10",
        "RPL10A", "RPL11", "RPL12", "RPL13", "RPL14", "RPL18", "RPL19",
        "RPL21", "RPL23", "RPL24", "RPL26", "RPL27A", "RPL30", "RPL31",
        "RPL32", "RPL34", "RPL36", "RPL37", "RPL37A", "RPLP2"
    ],
    "translation_factors": [
        "EIF2B2", "EIF2B4", "EIF2B5", "EIF2S1", "EIF3A", "EIF3E",
        "EIF3H", "EIF3M", "EIF4E", "EIF4G1", "EIF4G2", "EEF2"
    ],
    "splicing_factors": [
        "LSM2", "LSM5", "LSM6", "SF3A2", "SF3B2", "SF3B3", "SF3B4",
        "PRPF4B", "PRPF6", "PRPF18", "CRNKL1", "SNRPB", "SNRPG",
        "U2SURP", "HNRNPC", "SFPQ", "PHF5A", "ZMAT2", "USP39",
        "WBP11", "SON", "NSRP1", "SLU7", "SART1"
    ],
    "rna_processing_helicases": [
        "DDX24", "DDX41", "DDX46", "DHX15", "DHX16", "DHX36",
        "DHX37", "EXOSC2", "SRRT", "PHAX", "FTSJ3", "TRMT112",
        "YTHDC1"
    ],
    "proteasome_subunits": [
        "PSMA3", "PSMA4", "PSMB5", "PSMB6", "PSMB7", "PSMC1",
        "PSMC2", "PSMC4", "PSMD1", "PSMD4", "PSMD6", "PSMD7",
        "PSMD12"
    ],
    "ubiquitination_factors": [
        "UBA1", "UBL5", "SKP2", "COPS4", "COPS6", "COPS8"
    ],
    "rna_polymerase_subunits": [
        "POLR1E", "POLR2A", "POLR2D", "POLR2E", "POLR2F",
        "POLR2I", "POLR3E", "POLRMT"
    ],
    "mediator_complex_components": [
        "MED1", "MED6", "MED7", "MED10", "MED11", "MED12",
        "MED19", "MED22", "MED30"
    ],
    "transcription_regulators": [
        "CNOT1", "CNOT3", "GTF2H1", "CCNK", "PAF1", "CTR9",
        "NELFCD", "RPAP2", "SIN3A", "SUPT5H", "SUPT6H",
        "BDP1", "ARGLU1", "KAT8", "NFRKB"
    ],
    "chromatin_remodeling": [
        "TRRAP", "ACTR8", "RUVBL1", "BAP1", "ANKRD11",
        "SSRP1", "SMC4", "YEATS2", "YEATS4"
    ],
    "cell_cycle_regulators": [
        "CDK1", "CHEK1", "CDC27", "RCC1", "SKP2", "CEP192",
        "TPX2", "NUMA1", "NDC80", "NUF2", "DSN1", "BUB3",
        "CENPJ", "MZT1"
    ],
    "dna_replication_repair": [
        "POLA1", "POLA2", "POLD1", "RFC3", "RPA1", "RPA2",
        "RPA3", "RAD51", "DONSON", "ORC1", "RRM2"
    ],
    "nuclear_pore_complex": [
        "NUP107", "NUP160", "NUP54", "NUP98", "KPNB1",
        "XPO1", "CSE1L"
    ],
    "chaperones_heat_shock_proteins": [
        "HSPA5", "HSPA8", "HSPA9", "CCT3", "CCT4",
        "DNAJC19", "UXT"
    ],
    "mitochondrial_function": [
        "GFM1", "MRPL14", "MRPS35", "GFER", "PHB2", "PTCD1",
        "TIMM23B", "TIMM9", "OXA1L", "ISCA2", "HARS",
        "IARS2", "VARS", "QARS", "RARS", "DLD", "HMGCS1"
    ],
    "ribosome_biogenesis": [
        "GNL2", "KRR1", "RRP12", "NOL11", "NIP7", "LTV1",
        "MAK16", "WDR12", "WDR3", "WDR36", "NLE1",
        "TSR2", "SDAD1"
    ],
    "vesicle_transport": [
        "COPA", "COPZ1", "COG2", "VPS54", "GOLT1B"
    ],
    "rna_modification": [
        "FTSJ3", "TRMT112", "YTHDC1", "DNTTIP2"
    ],
    "signal_transduction": [
        "GAB2", "GPS1"
    ],
    "apoptosis_regulation": [
        "CASP8AP2", "CCDC86"
    ],
    "metabolism": [
        "DHODH",  # Pyrimidine biosynthesis
        "HMGCS1", # Cholesterol biosynthesis
        "DLD"     # Energy metabolism
    ],
    "transcription_rna_processing": [
        "AATF", "MAGOH", "NACA", "NIFK", "NCBP2", "SRRT"
    ],
    "chromatin_dna_interaction": [
        "WDR26", "YEATS2", "YEATS4", "NFRKB"
    ],
    "cell_cycle_division": [
        "FAM32A", "TPX2", "DONSON"
    ],
    "splicing_rna_binding": [
        "RBM14", "RBMX2", "SNRPB", "SNRPG", "U2SURP"
    ],
    "protein_transport_localization": [
        "KPNB1", "XPO1", "NDC80", "NUMA1"
    ],
    "protein_modification": [
        "PHF5A", "PIAS4"
    ],
    "unknown_multiple_roles": [
        "ABCE1", "ABCF1", "ACTR8", "ARGLU1", "GUCD1",
        "ICE1", "PELO", "ZMAT2"
    ]
}
