-- `default`.teste definition

CREATE TABLE default.teste
(

    `id` String,

    `nome` Nullable(String),

    `laboratorio` Nullable(String),

    `codigo_os` Nullable(String),

    `data_nascimento` Nullable(String),

    `idade` Nullable(Float64),

    `sexo` Nullable(Float64),

    `cpf` Nullable(String),

    `medico` Nullable(String),

    `atendimento` Nullable(String),

    `convenio` Nullable(String),

    `quantidade_exames` Nullable(Float64),

    `ALBUMINA` Nullable(Float64),

    `ALBUMINA_ref` Nullable(String),

    `ALBUMINA_status` Nullable(String),

    `ALFA_1_GLOBULINA` Nullable(Float64),

    `ALFA_1_GLOBULINA_ref` Nullable(String),

    `ALFA_1_GLOBULINA_status` Nullable(String),

    `ALFA_2_GLOBULINA` Nullable(Float64),

    `ALFA_2_GLOBULINA_ref` Nullable(String),

    `ALFA_2_GLOBULINA_status` Nullable(String),

    `AMILASE` Nullable(Float64),

    `AMILASE_ref` Nullable(String),

    `AMILASE_status` Nullable(String),

    `ATIVIDADE_DE_PROTROMBINA` Nullable(Float64),

    `ATIVIDADE_DE_PROTROMBINA_ref` Nullable(String),

    `ATIVIDADE_DE_PROTROMBINA_status` Nullable(String),

    `BACTÉRIAS_URINA` Nullable(String),

    `BACTÉRIAS_URINA_ref` Nullable(String),

    `BACTERIAS_URINA_status` Nullable(String),

    `BASE_EXCESS` Nullable(Float64),

    `BASE_EXCESS_ref` Nullable(String),

    `BASE_EXCESS_status` Nullable(String),

    `BASTONETES` Nullable(Float64),

    `BASTONETES_ref` Nullable(String),

    `BASTONETES_status` Nullable(String),

    `BASÓFILOS` Nullable(Float64),

    `BASÓFILOS_ref` Nullable(String),

    `BASOFILOS_status` Nullable(String),

    `BETA_1_GLOBULINA` Nullable(Float64),

    `BETA_1_GLOBULINA_ref` Nullable(String),

    `BETA_1_GLOBULINA_status` Nullable(String),

    `BETA_2_GLOBULINA` Nullable(Float64),

    `BETA_2_GLOBULINA_ref` Nullable(String),

    `BETA_2_GLOBULINA_status` Nullable(String),

    `BILIRRUBINA_DIRETA` Nullable(Float64),

    `BILIRRUBINA_DIRETA_ref` Nullable(String),

    `BILIRRUBINA_DIRETA_status` Nullable(String),

    `BILIRRUBINA_INDIRETA` Nullable(Float64),

    `BILIRRUBINA_INDIRETA_ref` Nullable(String),

    `BILIRRUBINA_INDIRETA_status` Nullable(String),

    `BILIRRUBINA_TOTAL` Nullable(Float64),

    `BILIRRUBINA_TOTAL_ref` Nullable(String),

    `BILIRRUBINA_TOTAL_status` Nullable(String),

    `BILIRRUBINAS_URINA` Nullable(String),

    `BILIRRUBINAS_URINA_ref` Nullable(String),

    `BILIRRUBINAS_URINA_status` Nullable(String),

    `BLASTOS` Nullable(Float64),

    `BLASTOS_ref` Nullable(String),

    `BLASTOS_status` Nullable(String),

    `CARIÓTIPO` Nullable(Float64),

    `CARIÓTIPO_ref` Nullable(String),

    `CARIOTIPO_status` Nullable(String),

    `CD4_ABSOLUTO` Nullable(Float64),

    `CD4_ABSOLUTO_ref` Nullable(Float64),

    `CD4_ABSOLUTO_status` Nullable(String),

    `CD4_RELATIVO` Nullable(Float64),

    `CD4_RELATIVO_ref` Nullable(Float64),

    `CD4_RELATIVO_status` Nullable(String),

    `CEA` Nullable(Float64),

    `CEA_ref` Nullable(String),

    `CEA_status` Nullable(String),

    `CHCM` Nullable(Float64),

    `CHCM_ref` Nullable(String),

    `CHCM_status` Nullable(String),

    `CILINDROS` Nullable(String),

    `CILINDROS_ref` Nullable(String),

    `CILINDROS_status` Nullable(String),

    `COLESTEROL_HDL` Nullable(Float64),

    `COLESTEROL_HDL_ref` Nullable(String),

    `COLESTEROL_HDL_status` Nullable(String),

    `COLESTEROL_LDL` Nullable(Float64),

    `COLESTEROL_LDL_ref` Nullable(String),

    `COLESTEROL_LDL_status` Nullable(String),

    `COLESTEROL_NÃO_HDL` Nullable(Float64),

    `COLESTEROL_NÃO_HDL_ref` Nullable(String),

    `COLESTEROL_NAO_HDL_status` Nullable(String),

    `COLESTEROL_TOTAL` Nullable(Float64),

    `COLESTEROL_TOTAL_ref` Nullable(String),

    `COLESTEROL_TOTAL_status` Nullable(String),

    `COOMBS_DIRETO` Nullable(String),

    `COOMBS_DIRETO_ref` Nullable(String),

    `COOMBS_DIRETO_status` Nullable(String),

    `CORPOS_CETÔNICOS` Nullable(String),

    `CORPOS_CETÔNICOS_ref` Nullable(String),

    `CORPOS_CETONICOS_status` Nullable(String),

    `COVID_19_PCR` Nullable(String),

    `COVID_19_PCR_ref` Nullable(String),

    `COVID_19_PCR_status` Nullable(String),

    `CREATININA` Nullable(Float64),

    `CREATININA_ref` Nullable(String),

    `CREATININA_status` Nullable(String),

    `CRISTAIS` Nullable(String),

    `CRISTAIS_ref` Nullable(String),

    `CRISTAIS_status` Nullable(String),

    `CRYPTOCOCCUS_ANTÍGENO` Nullable(Float64),

    `CRYPTOCOCCUS_ANTÍGENO_ref` Nullable(Float64),

    `CRYPTOCOCCUS_ANTIGENO_status` Nullable(String),

    `CULTURA_DE_SANGUE` Nullable(String),

    `CULTURA_DE_SANGUE_ref` Nullable(String),

    `CULTURA_DE_SANGUE_status` Nullable(String),

    `CULTURA_DE_URINA` Nullable(String),

    `CULTURA_DE_URINA_ref` Nullable(String),

    `CULTURA_DE_URINA_status` Nullable(String),

    `CÁLCIO_IÔNICO` Nullable(Float64),

    `CÁLCIO_IÔNICO_ref` Nullable(String),

    `CALCIO_IONICO_status` Nullable(String),

    `CÉLULAS_EPITELIAIS` Nullable(Float64),

    `CÉLULAS_EPITELIAIS_ref` Nullable(String),

    `CELULAS_EPITELIAIS_status` Nullable(String),

    `DENGUE_NS1` Nullable(String),

    `DENGUE_NS1_ref` Nullable(String),

    `DENGUE_NS1_status` Nullable(String),

    `DENSIDADE_URINA` Nullable(Float64),

    `DENSIDADE_URINA_ref` Nullable(String),

    `DENSIDADE_URINA_status` Nullable(String),

    `EOSINÓFILOS` Nullable(Float64),

    `EOSINÓFILOS_ref` Nullable(String),

    `EOSINOFILOS_status` Nullable(String),

    `FATOR_RH` Nullable(String),

    `FATOR_RH_ref` Nullable(String),

    `FATOR_RH_status` Nullable(String),

    `FERRITINA` Nullable(Float64),

    `FERRITINA_ref` Nullable(String),

    `FERRITINA_status` Nullable(String),

    `FERRO` Nullable(Float64),

    `FERRO_SÉRICO` Nullable(Float64),

    `FERRO_SÉRICO_ref` Nullable(String),

    `FERRO_SERICO_status` Nullable(String),

    `FERRO_ref` Nullable(Float64),

    `FERRO_status` Nullable(String),

    `FOSFATASE_ALCALINA` Nullable(Float64),

    `FOSFATASE_ALCALINA_ref` Nullable(String),

    `FOSFATASE_ALCALINA_status` Nullable(String),

    `GAMA_GLOBULINA` Nullable(Float64),

    `GAMA_GLOBULINA_ref` Nullable(String),

    `GAMA_GLOBULINA_status` Nullable(String),

    `GAMA_GT` Nullable(Float64),

    `GAMA_GT_ref` Nullable(String),

    `GAMA_GT_status` Nullable(String),

    `GLICOSE` Nullable(Float64),

    `GLICOSE_ref` Nullable(String),

    `GLICOSE_status` Nullable(String),

    `GRUPO_ABO` Nullable(String),

    `GRUPO_ABO_ref` Nullable(String),

    `GRUPO_ABO_status` Nullable(String),

    `HCG` Nullable(Float64),

    `HCG_ref` Nullable(String),

    `HCG_status` Nullable(String),

    `HCM` Nullable(Float64),

    `HCM_ref` Nullable(String),

    `HCM_status` Nullable(String),

    `HCO3` Nullable(Float64),

    `HCO3_ref` Nullable(String),

    `HCO3_status` Nullable(String),

    `HEMATÓCRITO` Nullable(Float64),

    `HEMATÓCRITO_ref` Nullable(String),

    `HEMATOCRITO_status` Nullable(String),

    `HEMOGLOBINA` Nullable(Float64),

    `HEMOGLOBINA_URINA` Nullable(Float64),

    `HEMOGLOBINA_URINA_ref` Nullable(Float64),

    `HEMOGLOBINA_URINA_status` Nullable(String),

    `HEMOGLOBINA_ref` Nullable(String),

    `HEMOGLOBINA_status` Nullable(String),

    `HEMÁCIAS` Nullable(String),

    `HEMÁCIAS_URINA` Nullable(String),

    `HEMÁCIAS_URINA_ref` Nullable(String),

    `HEMACIAS_URINA_status` Nullable(String),

    `HEMÁCIAS_ref` Nullable(String),

    `HEMACIAS_status` Nullable(String),

    `HbA1c` Nullable(Float64),

    `HbA1c_ref` Nullable(String),

    `HbA1c_status` Nullable(String),

    `INR` Nullable(Float64),

    `INR_ref` Nullable(String),

    `INR_status` Nullable(String),

    `ISI` Nullable(Float64),

    `ISI_ref` Nullable(String),

    `ISI_status` Nullable(String),

    `IgE_TOTAL` Nullable(Float64),

    `IgE_TOTAL_ref` Nullable(String),

    `IgE_TOTAL_status` Nullable(String),

    `LACTATO` Nullable(Float64),

    `LACTATO_ref` Nullable(String),

    `LACTATO_status` Nullable(String),

    `LEUCÓCITOS` Nullable(String),

    `LEUCÓCITOS_URINA` Nullable(String),

    `LEUCÓCITOS_URINA_ref` Nullable(String),

    `LEUCOCITOS_URINA_status` Nullable(String),

    `LEUCÓCITOS_ref` Nullable(String),

    `LEUCOCITOS_status` Nullable(String),

    `LINFÓCITOS_ATÍPICOS` Nullable(Float64),

    `LINFÓCITOS_ATÍPICOS_ref` Nullable(String),

    `LINFOCITOS_ATIPICOS_status` Nullable(String),

    `LINFÓCITOS_TÍPICOS` Nullable(Float64),

    `LINFÓCITOS_TÍPICOS_ref` Nullable(String),

    `LINFOCITOS_TIPICOS_status` Nullable(String),

    `LIPASE` Nullable(Float64),

    `LIPASE_ref` Nullable(String),

    `LIPASE_status` Nullable(String),

    `MAGNÉSIO` Nullable(Float64),

    `MAGNÉSIO_ref` Nullable(String),

    `MAGNESIO_status` Nullable(String),

    `METAMIELÓCITOS` Nullable(Float64),

    `METAMIELÓCITOS_ref` Nullable(String),

    `METAMIELOCITOS_status` Nullable(String),

    `MIELÓCITOS` Nullable(Float64),

    `MIELÓCITOS_ref` Nullable(String),

    `MIELOCITOS_status` Nullable(String),

    `MONÓCITOS` Nullable(Float64),

    `MONÓCITOS_ref` Nullable(String),

    `MONÓCITOS_status` Nullable(String),

    `NEUTRÓFILOS_TOTAIS` Nullable(Float64),

    `NEUTRÓFILOS_TOTAIS_ref` Nullable(String),

    `NEUTRÓFILOS_TOTAIS_status` Nullable(String),

    `NITRITOS` Nullable(String),

    `NITRITOS_ref` Nullable(String),

    `NITRITOS_status` Nullable(String),

    `O2_SATURAÇÃO` Nullable(Float64),

    `O2_SATURAÇÃO_ref` Nullable(String),

    `O2_SATURAÇÃO_status` Nullable(String),

    `PARATORMÔNIO` Nullable(Float64),

    `PARATORMÔNIO_ref` Nullable(String),

    `PARATORMÔNIO_status` Nullable(String),

    `PCO2` Nullable(Float64),

    `PCO2_ref` Nullable(String),

    `PCO2_status` Nullable(String),

    `PCR` Nullable(Float64),

    `PCR_ref` Nullable(String),

    `PCR_status` Nullable(String),

    `PH_URINA` Nullable(Float64),

    `PH_URINA_ref` Nullable(String),

    `PH_URINA_status` Nullable(String),

    `PLAQUETAS` Nullable(Float64),

    `PLAQUETAS_ref` Nullable(String),

    `PLAQUETAS_status` Nullable(String),

    `PO2` Nullable(Float64),

    `PO2_ref` Nullable(String),

    `PO2_status` Nullable(String),

    `POTÁSSIO` Nullable(Float64),

    `POTÁSSIO_ref` Nullable(String),

    `POTÁSSIO_status` Nullable(String),

    `PROGESTERONA` Nullable(Float64),

    `PROGESTERONA_ref` Nullable(String),

    `PROGESTERONA_status` Nullable(String),

    `PROLACTINA` Nullable(Float64),

    `PROLACTINA_ref` Nullable(Float64),

    `PROLACTINA_status` Nullable(String),

    `PROMIELÓCITOS` Nullable(Float64),

    `PROMIELÓCITOS_ref` Nullable(String),

    `PROMIELÓCITOS_status` Nullable(String),

    `PROTEÍNAS_TOTAIS` Nullable(Float64),

    `PROTEÍNAS_TOTAIS_ref` Nullable(String),

    `PROTEÍNAS_TOTAIS_status` Nullable(String),

    `PSA_LIVRE` Nullable(Float64),

    `PSA_LIVRE_ref` Nullable(String),

    `PSA_LIVRE_status` Nullable(String),

    `PSA_TOTAL` Nullable(Float64),

    `PSA_TOTAL_ref` Nullable(String),

    `PSA_TOTAL_status` Nullable(String),

    `RDW` Nullable(Float64),

    `RDW_ref` Nullable(String),

    `RDW_status` Nullable(String),

    `RELAÇÃO_A_G` Nullable(Float64),

    `RELAÇÃO_A_G_ref` Nullable(String),

    `RELAÇÃO_A_G_status` Nullable(String),

    `RELAÇÃO_PSA_LIVRE_TOTAL` Nullable(Float64),

    `RELAÇÃO_PSA_LIVRE_TOTAL_ref` Nullable(String),

    `RELAÇÃO_PSA_LIVRE_TOTAL_status` Nullable(String),

    `RELAÇÃO_TP` Nullable(Float64),

    `RELAÇÃO_TP_ref` Nullable(String),

    `RELAÇÃO_TP_status` Nullable(String),

    `SANGUE_OCULTO` Nullable(String),

    `SANGUE_OCULTO_ref` Nullable(String),

    `SANGUE_OCULTO_status` Nullable(String),

    `SEGMENTADOS` Nullable(Float64),

    `SEGMENTADOS_ref` Nullable(String),

    `SEGMENTADOS_status` Nullable(String),

    `SÓDIO` Nullable(Float64),

    `SÓDIO_ref` Nullable(String),

    `SÓDIO_status` Nullable(String),

    `T4_LIVRE` Nullable(Float64),

    `T4_LIVRE_ref` Nullable(String),

    `T4_LIVRE_status` Nullable(String),

    `TESTOSTERONA_TOTAL` Nullable(Float64),

    `TESTOSTERONA_TOTAL_ref` Nullable(Float64),

    `TESTOSTERONA_TOTAL_status` Nullable(String),

    `TGO` Nullable(Float64),

    `TGO_ref` Nullable(String),

    `TGO_status` Nullable(String),

    `TGP` Nullable(Float64),

    `TGP_ref` Nullable(String),

    `TGP_status` Nullable(String),

    `TP___TEMPO_PROTROMBINA__PACIENTE_` Nullable(Float64),

    `TP___TEMPO_PROTROMBINA__PACIENTE__ref` Nullable(String),

    `TP___TEMPO_PROTROMBINA__PACIENTE__status` Nullable(String),

    `TP___TEMPO_PROTROMBINA__PADRÃO_` Nullable(Float64),

    `TP___TEMPO_PROTROMBINA__PADRÃO__ref` Nullable(String),

    `TP___TEMPO_PROTROMBINA__PADRÃO__status` Nullable(String),

    `TRIGLICERÍDEOS` Nullable(Float64),

    `TRIGLICERÍDEOS_ref` Nullable(String),

    `TRIGLICERÍDEOS_status` Nullable(String),

    `TSH` Nullable(Float64),

    `TSH_ref` Nullable(String),

    `TSH_status` Nullable(String),

    `TTPA___PLASMA_CONTROLE` Nullable(Float64),

    `TTPA___PLASMA_CONTROLE_ref` Nullable(String),

    `TTPA___PLASMA_CONTROLE_status` Nullable(String),

    `TTPA___PLASMA_PACIENTE` Nullable(Float64),

    `TTPA___PLASMA_PACIENTE_ref` Nullable(String),

    `TTPA___PLASMA_PACIENTE_status` Nullable(String),

    `TTPA___RELAÇÃO` Nullable(Float64),

    `TTPA___RELAÇÃO_ref` Nullable(String),

    `TTPA___RELAÇÃO_status` Nullable(String),

    `UREIA` Nullable(Float64),

    `UREIA_ref` Nullable(String),

    `UREIA_status` Nullable(String),

    `UROCULTURA_UFC` Nullable(Float64),

    `UROCULTURA_UFC_ref` Nullable(String),

    `UROCULTURA_UFC_status` Nullable(String),

    `VCM` Nullable(Float64),

    `VCM_ref` Nullable(String),

    `VCM_status` Nullable(String),

    `VHS` Nullable(Float64),

    `VHS_ref` Nullable(String),

    `VHS_status` Nullable(String),

    `VITAMINA_B12` Nullable(Float64),

    `VITAMINA_B12_ref` Nullable(String),

    `VITAMINA_B12_status` Nullable(String),

    `VITAMINA_D` Nullable(Float64),

    `VITAMINA_D_ref` Nullable(String),

    `VITAMINA_D_status` Nullable(String),

    `VPM` Nullable(Float64),

    `VPM_ref` Nullable(String),

    `VPM_status` Nullable(String),

    `ZINCO` Nullable(Float64),

    `ZINCO_ref` Nullable(Float64),

    `ZINCO_status` Nullable(String),

    `eTFG` Nullable(Float64),

    `eTFG_ref` Nullable(String),

    `eTFG_status` Nullable(String),

    `ÁCIDO_FÓLICO` Nullable(Float64),

    `ÁCIDO_FÓLICO_ref` Nullable(String),

    `ÁCIDO_FÓLICO_status` Nullable(String),

    `ÁCIDO_ÚRICO` Nullable(Float64),

    `ÁCIDO_ÚRICO_ref` Nullable(String),

    `ÁCIDO_ÚRICO_status` Nullable(String),

    `DOC_CARTEIRA_xml` Nullable(String),

    `DOC_COD_xml` Nullable(String),

    `DOC_EXAMES_xml` Nullable(String),

    `DOC_GUIA_xml` Nullable(String),

    `DOC_IDDOCOP_xml` Nullable(String),

    `DOC_PS_xml` Nullable(UInt32),

    `ALUMÍNIO` Nullable(Float64),

    `ALUMÍNIO_ref` Nullable(Float64),

    `ALUMÍNIO_status` Nullable(String),

    `CAXUMBA_IgG` Nullable(Float64),

    `CAXUMBA_IgG_ref` Nullable(Float64),

    `CAXUMBA_IgG_status` Nullable(String),

    `CROMO` Nullable(Float64),

    `CROMO_ref` Nullable(Float64),

    `CROMO_status` Nullable(String),

    `FSH` Nullable(Float64),

    `FSH_ref` Nullable(Float64),

    `FSH_status` Nullable(String),

    `HOMA_IR` Nullable(Float64),

    `HOMA_IR_ref` Nullable(Float64),

    `HOMA_IR_status` Nullable(String),

    `INSULINA` Nullable(Float64),

    `INSULINA_ref` Nullable(Float64),

    `INSULINA_status` Nullable(String),

    `LH` Nullable(Float64),

    `LH_ref` Nullable(Float64),

    `LH_status` Nullable(String),

    `MERCÚRIO` Nullable(Float64),

    `MERCÚRIO_ref` Nullable(Float64),

    `MERCÚRIO_status` Nullable(String),

    `MICROALBUMINÚRIA` Nullable(Float64),

    `MICROALBUMINÚRIA_ref` Nullable(Float64),

    `MICROALBUMINÚRIA_status` Nullable(String),

    `T3` Nullable(Float64),

    `T3_ref` Nullable(Float64),

    `T3_status` Nullable(String),

    `T4` Nullable(Float64),

    `T4_ref` Nullable(Float64),

    `T4_status` Nullable(String),

    `VITAMINA_A` Nullable(Float64),

    `VITAMINA_A_ref` Nullable(Float64),

    `VITAMINA_A_status` Nullable(String),

    `VITAMINA_D3` Nullable(Float64),

    `VITAMINA_D3_ref` Nullable(Float64),

    `VITAMINA_D3_status` Nullable(String),

    `ANTI_TRANSGLUTAMINASE_IgA` Nullable(Float64),

    `ANTI_TRANSGLUTAMINASE_IgA_ref` Nullable(Float64),

    `ANTI_TRANSGLUTAMINASE_IgA_status` Nullable(String),

    `CARBOXIHEMOGLOBINA` Nullable(Float64),

    `CARBOXIHEMOGLOBINA_ref` Nullable(Float64),

    `CARBOXIHEMOGLOBINA_status` Nullable(String),

    `CHUMBO` Nullable(Float64),

    `CHUMBO_ref` Nullable(Float64),

    `CHUMBO_status` Nullable(String),

    `COBRE` Nullable(Float64),

    `COBRE_ref` Nullable(Float64),

    `COBRE_status` Nullable(String),

    `ESTRADIOL` Nullable(Float64),

    `ESTRADIOL_ref` Nullable(Float64),

    `ESTRADIOL_status` Nullable(String),

    `HOMA_BETA` Nullable(Float64),

    `HOMA_BETA_ref` Nullable(Float64),

    `HOMA_BETA_status` Nullable(String),

    `IgA` Nullable(Float64),

    `IgA_ref` Nullable(Float64),

    `IgA_status` Nullable(String),

    `IgG` Nullable(Float64),

    `IgG_ref` Nullable(Float64),

    `IgG_status` Nullable(String),

    `IgM` Nullable(Float64),

    `IgM_ref` Nullable(Float64),

    `IgM_status` Nullable(String),

    `MANGANÊS` Nullable(Float64),

    `MANGANÊS_ref` Nullable(Float64),

    `MANGANÊS_status` Nullable(String),

    `LINFÓCITOS` Nullable(Float64),

    `LINFÓCITOS_ref` Nullable(Float64),

    `LINFÓCITOS_status` Nullable(String),

    `TESTOSTERONA_LIVRE` Nullable(Float64),

    `TESTOSTERONA_LIVRE_ref` Nullable(Float64),

    `TESTOSTERONA_LIVRE_status` Nullable(String),

    `dados_parceiro_chave_publica` Nullable(String),

    `PRODUTOCODIGO` Nullable(String),

    `PRODUTONOME` Nullable(String),

    `CONTRATOCODIGO` Nullable(String),

    `CONTRATONOME` Nullable(String),

    `CONTRATOTPEMPRESA` Nullable(String),

    `REDE_ATEND` Nullable(String),

    `PLANONOME` Nullable(String),

    `INICIO_CONTRATO` Nullable(DateTime64(3)),

    `FIM_CONTRATO` Nullable(String),

    `GRUPO_EMPRESA_HRP_ATEND` Nullable(String)
)
ENGINE = MergeTree
ORDER BY assumeNotNull(codigo_os)
SETTINGS index_granularity = 8192;