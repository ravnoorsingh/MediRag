# Sample FHIR Data for Clinical Decision Support System

This directory contains sample FHIR R4 compliant JSON files that can be uploaded to the Clinical Decision Support System for testing and demonstration purposes.

## 📋 Available Sample Cases

### 1. **patient_diabetes_hypertension.json**
**Patient**: Robert Michael Johnson (Male, 60 years)
- **Conditions**: Type 2 Diabetes Mellitus, Essential Hypertension
- **Medications**: Metformin 1000mg, Lisinopril 10mg
- **Allergies**: Penicillin (moderate urticaria)
- **Recent Labs**: HbA1c 7.8% (elevated), BP 152/94 mmHg (elevated)
- **Clinical Scenarios**: 
  - Diabetes management optimization
  - Hypertension treatment adjustment
  - Cardiovascular risk assessment

### 2. **patient_cardiac_case.json**
**Patient**: Maria Elena Martinez (Female, 66 years)
- **Conditions**: Coronary Artery Disease, Atrial Fibrillation
- **Medications**: Warfarin 5mg, Metoprolol 25mg
- **Allergies**: Aspirin (severe bronchospasm)
- **Recent Tests**: INR 2.8 (therapeutic), ECG showing AFib
- **Clinical Scenarios**:
  - Anticoagulation management
  - Cardiac rhythm control
  - Drug allergy considerations

### 3. **patient_respiratory_case.json**
**Patient**: James William Thompson (Male, 50 years)
- **Conditions**: Asthma (childhood onset), COPD
- **Medications**: Albuterol inhaler (PRN), Advair Diskus 250/50
- **Allergies**: Grass pollen (environmental)
- **Recent Tests**: Peak flow 320 L/min (low), O2 sat 94% (low)
- **Clinical Scenarios**:
  - Respiratory symptom management
  - Inhaler technique optimization
  - COPD exacerbation prevention

### 4. **patient_elderly_multiple_conditions.json**
**Patient**: Dorothy Mae Wilson (Female, 86 years)
- **Conditions**: Alzheimer's Disease, Osteoporosis, CKD Stage 3a
- **Medications**: Donepezil 10mg, Alendronate 70mg weekly
- **Allergies**: Sulfa drugs (skin rash)
- **Recent Tests**: MMSE 18/30 (cognitive decline), Creatinine 1.8 mg/dL (elevated)
- **Clinical Scenarios**:
  - Geriatric polypharmacy management
  - Cognitive decline monitoring
  - Bone health in CKD patients

## 🚀 How to Use These Files

### **Method 1: Web Interface Upload**
1. Start the Clinical Decision Support System:
   ```bash
   cd /Users/ravnoorsingh/Downloads/MediRag
   ./run_app.sh
   ```
2. Open your browser to `http://localhost:5001`
3. Click "Upload Patient Data"
4. Select "Upload FHIR JSON File"
5. Choose one of the sample files from this directory
6. Click "Upload and Process"

### **Method 2: Direct File Upload**
- Navigate to the patient upload page
- Drag and drop any `.json` file from this directory
- The system will automatically parse and display patient information

## 🔍 Testing Clinical Queries

After uploading a patient file, try these example clinical questions:

### **For Diabetes/Hypertension Patient:**
- "What are the best evidence-based treatment options for this patient's diabetes management?"
- "Should we adjust the blood pressure medication given the current readings?"
- "What additional tests should be ordered for cardiovascular risk assessment?"

### **For Cardiac Patient:**
- "Is the current INR level appropriate for this patient's atrial fibrillation?"
- "What are safe antiplatelet alternatives given the aspirin allergy?"
- "Should we consider rhythm control vs rate control strategies?"

### **For Respiratory Patient:**
- "What interventions can improve this patient's peak flow measurements?"
- "Are there any medication interactions between asthma and COPD treatments?"
- "What is the prognosis for patients with both asthma and COPD?"

### **For Elderly Patient:**
- "How should we adjust medications considering the kidney disease?"
- "What are the risks of continuing alendronate in this elderly patient?"
- "What additional monitoring is needed for dementia progression?"

## 📊 Expected System Behavior

When you upload these files, the system will:

1. **Parse FHIR Resources**: Extract Patient, Condition, Medication, Allergy, and Observation data
2. **Display Patient Summary**: Show demographics, conditions, medications, and recent test results
3. **Enable Clinical Queries**: Allow you to ask specific clinical questions
4. **Generate Evidence-Based Recommendations**: Provide 3 care options with:
   - Clinical rationale
   - Evidence citations from medical literature
   - Contraindications and monitoring requirements
   - Expected outcomes

## ✅ Data Validation

All sample files are:
- **FHIR R4 Compliant**: Follow HL7 FHIR specification
- **Clinically Realistic**: Based on common medical scenarios
- **Comprehensive**: Include multiple resource types for testing
- **Validated**: Tested with the Clinical Decision Support System

## 🔧 Customization

To create your own test cases:
1. Follow the FHIR R4 Bundle structure
2. Include Patient resource as the first entry
3. Add Condition, MedicationRequest, AllergyIntolerance, and Observation resources
4. Reference the Patient ID in all other resources
5. Use standard medical coding systems (SNOMED CT, ICD-10, LOINC, RxNorm)

## 📚 Additional Resources

- [FHIR R4 Specification](https://hl7.org/fhir/R4/)
- [FHIR Resource Examples](https://hl7.org/fhir/R4/resourcelist.html)
- [Medical Terminology Systems](https://www.nlm.nih.gov/research/umls/)

---

**Note**: These are synthetic patient records created for testing purposes only. No real patient data is included in these files.