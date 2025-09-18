"""
FHIR Data Parser for Clinical Decision Support System
Handles parsing and extraction of patient data from FHIR R4 format and other common healthcare data formats

This module provides:
1. FHIR R4 resource parsing (Patient, Condition, Medication, Observation, etc.)
2. Data extraction and normalization for clinical decision making
3. Conversion to internal patient representation
4. Support for both FHIR bundles and individual resources
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class FHIRResourceType(Enum):
    """Supported FHIR resource types"""
    PATIENT = "Patient"
    CONDITION = "Condition"
    MEDICATION = "Medication"
    MEDICATION_REQUEST = "MedicationRequest"
    OBSERVATION = "Observation"
    ENCOUNTER = "Encounter"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    PROCEDURE = "Procedure"


@dataclass
class ParsedPatientData:
    """Standardized patient data structure for clinical decision making"""
    patient_id: str
    demographics: Dict[str, Any]
    conditions: List[Dict[str, Any]]
    medications: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    allergies: List[Dict[str, Any]]
    procedures: List[Dict[str, Any]]
    encounters: List[Dict[str, Any]]
    raw_fhir: Optional[Dict[str, Any]] = None


class FHIRDataParser:
    """
    Parser for FHIR R4 healthcare data format
    
    Converts complex FHIR resources into normalized patient data
    suitable for clinical decision support systems
    """
    
    def __init__(self):
        """Initialize the FHIR parser"""
        self.supported_resources = [resource.value for resource in FHIRResourceType]
        print("📋 FHIR Data Parser initialized")
    
    def parse_fhir_bundle(self, fhir_bundle: Dict[str, Any]) -> ParsedPatientData:
        """
        Parse a complete FHIR Bundle containing patient resources
        
        Args:
            fhir_bundle: FHIR Bundle resource containing patient data
            
        Returns:
            ParsedPatientData: Normalized patient data structure
        """
        print("🔍 Parsing FHIR Bundle...")
        
        # Extract entries from bundle
        entries = fhir_bundle.get('entry', [])
        if not entries:
            raise ValueError("Empty FHIR Bundle provided")
        
        # Organize resources by type
        resources_by_type = {}
        for entry in entries:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            
            if resource_type in self.supported_resources:
                if resource_type not in resources_by_type:
                    resources_by_type[resource_type] = []
                resources_by_type[resource_type].append(resource)
        
        # Parse each resource type
        patient_data = self._parse_resources(resources_by_type)
        patient_data.raw_fhir = fhir_bundle
        
        print(f"✅ Parsed FHIR Bundle: {len(entries)} resources processed")
        return patient_data
    
    def parse_fhir_resources(
        self, 
        resources: List[Dict[str, Any]]
    ) -> ParsedPatientData:
        """
        Parse individual FHIR resources
        
        Args:
            resources: List of FHIR resources
            
        Returns:
            ParsedPatientData: Normalized patient data
        """
        print(f"🔍 Parsing {len(resources)} FHIR resources...")
        
        # Group resources by type
        resources_by_type = {}
        for resource in resources:
            resource_type = resource.get('resourceType')
            if resource_type in self.supported_resources:
                if resource_type not in resources_by_type:
                    resources_by_type[resource_type] = []
                resources_by_type[resource_type].append(resource)
        
        # Parse organized resources
        patient_data = self._parse_resources(resources_by_type)
        
        print(f"✅ Parsed {len(resources)} FHIR resources")
        return patient_data
    
    def _parse_resources(self, resources_by_type: Dict[str, List[Dict]]) -> ParsedPatientData:
        """Parse organized FHIR resources into standardized format"""
        
        # Parse patient demographics (required)
        patient_resources = resources_by_type.get('Patient', [])
        if not patient_resources:
            raise ValueError("No Patient resource found in FHIR data")
        
        patient_resource = patient_resources[0]  # Use first patient resource
        patient_id = patient_resource.get('id', 'unknown')
        demographics = self._parse_patient_demographics(patient_resource)
        
        # Parse conditions
        conditions = []
        for condition in resources_by_type.get('Condition', []):
            parsed_condition = self._parse_condition(condition)
            if parsed_condition:
                conditions.append(parsed_condition)
        
        # Parse medications
        medications = []
        for med_resource in resources_by_type.get('MedicationRequest', []):
            parsed_med = self._parse_medication_request(med_resource)
            if parsed_med:
                medications.append(parsed_med)
        
        # Parse observations (labs, vitals, etc.)
        observations = []
        for obs in resources_by_type.get('Observation', []):
            parsed_obs = self._parse_observation(obs)
            if parsed_obs:
                observations.append(parsed_obs)
        
        # Parse allergies
        allergies = []
        for allergy in resources_by_type.get('AllergyIntolerance', []):
            parsed_allergy = self._parse_allergy(allergy)
            if parsed_allergy:
                allergies.append(parsed_allergy)
        
        # Parse procedures
        procedures = []
        for procedure in resources_by_type.get('Procedure', []):
            parsed_procedure = self._parse_procedure(procedure)
            if parsed_procedure:
                procedures.append(parsed_procedure)
        
        # Parse encounters
        encounters = []
        for encounter in resources_by_type.get('Encounter', []):
            parsed_encounter = self._parse_encounter(encounter)
            if parsed_encounter:
                encounters.append(parsed_encounter)
        
        return ParsedPatientData(
            patient_id=patient_id,
            demographics=demographics,
            conditions=conditions,
            medications=medications,
            observations=observations,
            allergies=allergies,
            procedures=procedures,
            encounters=encounters
        )
    
    def _parse_patient_demographics(self, patient_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Patient resource for demographics"""
        demographics = {}
        
        # Basic identifiers
        demographics['id'] = patient_resource.get('id', 'unknown')
        
        # Name parsing
        names = patient_resource.get('name', [])
        if names:
            name = names[0]  # Use first name
            given_names = name.get('given', [])
            family_name = name.get('family', '')
            
            demographics['first_name'] = given_names[0] if given_names else ''
            demographics['last_name'] = family_name
            demographics['full_name'] = f"{' '.join(given_names)} {family_name}".strip()
        
        # Gender
        demographics['gender'] = patient_resource.get('gender', 'unknown')
        
        # Birth date and age calculation
        birth_date = patient_resource.get('birthDate')
        if birth_date:
            demographics['birth_date'] = birth_date
            demographics['age'] = self._calculate_age(birth_date)
        
        # Contact information
        telecoms = patient_resource.get('telecom', [])
        for telecom in telecoms:
            system = telecom.get('system')
            value = telecom.get('value')
            if system == 'phone':
                demographics['phone'] = value
            elif system == 'email':
                demographics['email'] = value
        
        # Address
        addresses = patient_resource.get('address', [])
        if addresses:
            address = addresses[0]
            demographics['address'] = {
                'street': ' '.join(address.get('line', [])),
                'city': address.get('city', ''),
                'state': address.get('state', ''),
                'postal_code': address.get('postalCode', ''),
                'country': address.get('country', '')
            }
        
        return demographics
    
    def _parse_condition(self, condition_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse FHIR Condition resource"""
        try:
            condition = {}
            
            # Condition ID
            condition['id'] = condition_resource.get('id', '')
            
            # Condition code and display
            code = condition_resource.get('code', {})
            codings = code.get('coding', [])
            if codings:
                primary_coding = codings[0]
                condition['code'] = primary_coding.get('code', '')
                condition['system'] = primary_coding.get('system', '')
                condition['display'] = primary_coding.get('display', '')
            
            condition['text'] = code.get('text', condition.get('display', ''))
            
            # Clinical status
            clinical_status = condition_resource.get('clinicalStatus', {})
            status_codings = clinical_status.get('coding', [])
            if status_codings:
                condition['status'] = status_codings[0].get('code', 'unknown')
            
            # Onset information
            onset_datetime = condition_resource.get('onsetDateTime')
            if onset_datetime:
                condition['onset_date'] = onset_datetime
                condition['duration'] = self._calculate_duration(onset_datetime)
            
            # Severity
            severity = condition_resource.get('severity', {})
            if severity:
                severity_codings = severity.get('coding', [])
                if severity_codings:
                    condition['severity'] = severity_codings[0].get('display', 'unknown')
            
            return condition
            
        except Exception as e:
            print(f"Warning: Could not parse condition: {e}")
            return None
    
    def _parse_medication_request(self, med_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse FHIR MedicationRequest resource"""
        try:
            medication = {}
            
            # Medication ID
            medication['id'] = med_resource.get('id', '')
            
            # Medication code
            medication_code = med_resource.get('medicationCodeableConcept', {})
            codings = medication_code.get('coding', [])
            if codings:
                primary_coding = codings[0]
                medication['code'] = primary_coding.get('code', '')
                medication['system'] = primary_coding.get('system', '')
                medication['name'] = primary_coding.get('display', '')
            
            medication['text'] = medication_code.get('text', medication.get('name', ''))
            
            # Dosage instructions
            dosage_instructions = med_resource.get('dosageInstruction', [])
            if dosage_instructions:
                dosage = dosage_instructions[0]
                medication['dosage_text'] = dosage.get('text', '')
                
                # Dose quantity
                dose = dosage.get('doseAndRate', [{}])[0].get('doseQuantity', {})
                if dose:
                    medication['dose_value'] = dose.get('value')
                    medication['dose_unit'] = dose.get('unit')
                
                # Frequency
                timing = dosage.get('timing', {})
                repeat = timing.get('repeat', {})
                if repeat:
                    medication['frequency'] = repeat.get('frequency', 1)
                    medication['period'] = repeat.get('period', 1)
                    medication['period_unit'] = repeat.get('periodUnit', 'day')
            
            # Status
            medication['status'] = med_resource.get('status', 'unknown')
            
            # Intent
            medication['intent'] = med_resource.get('intent', 'unknown')
            
            return medication
            
        except Exception as e:
            print(f"Warning: Could not parse medication: {e}")
            return None
    
    def _parse_observation(self, obs_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse FHIR Observation resource (labs, vitals, etc.)"""
        try:
            observation = {}
            
            # Observation ID
            observation['id'] = obs_resource.get('id', '')
            
            # Observation code
            code = obs_resource.get('code', {})
            codings = code.get('coding', [])
            if codings:
                primary_coding = codings[0]
                observation['code'] = primary_coding.get('code', '')
                observation['system'] = primary_coding.get('system', '')
                observation['name'] = primary_coding.get('display', '')
            
            observation['text'] = code.get('text', observation.get('name', ''))
            
            # Value
            value_quantity = obs_resource.get('valueQuantity')
            value_string = obs_resource.get('valueString')
            value_codeable = obs_resource.get('valueCodeableConcept')
            
            if value_quantity:
                observation['value'] = value_quantity.get('value')
                observation['unit'] = value_quantity.get('unit')
                observation['value_type'] = 'quantity'
            elif value_string:
                observation['value'] = value_string
                observation['value_type'] = 'string'
            elif value_codeable:
                codings = value_codeable.get('coding', [])
                if codings:
                    observation['value'] = codings[0].get('display', '')
                    observation['value_type'] = 'coded'
            
            # Status
            observation['status'] = obs_resource.get('status', 'unknown')
            
            # Effective date/time
            effective_datetime = obs_resource.get('effectiveDateTime')
            if effective_datetime:
                observation['effective_date'] = effective_datetime
            
            # Category
            categories = obs_resource.get('category', [])
            if categories:
                category_codings = categories[0].get('coding', [])
                if category_codings:
                    observation['category'] = category_codings[0].get('display', 'unknown')
            
            # Reference ranges
            reference_ranges = obs_resource.get('referenceRange', [])
            if reference_ranges:
                ref_range = reference_ranges[0]
                observation['reference_range'] = {
                    'low': ref_range.get('low', {}).get('value'),
                    'high': ref_range.get('high', {}).get('value'),
                    'unit': ref_range.get('low', {}).get('unit', ref_range.get('high', {}).get('unit'))
                }
            
            return observation
            
        except Exception as e:
            print(f"Warning: Could not parse observation: {e}")
            return None
    
    def _parse_allergy(self, allergy_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse FHIR AllergyIntolerance resource"""
        try:
            allergy = {}
            
            # Allergy ID
            allergy['id'] = allergy_resource.get('id', '')
            
            # Substance
            code = allergy_resource.get('code', {})
            codings = code.get('coding', [])
            if codings:
                primary_coding = codings[0]
                allergy['substance_code'] = primary_coding.get('code', '')
                allergy['substance_name'] = primary_coding.get('display', '')
            
            allergy['substance'] = code.get('text', allergy.get('substance_name', ''))
            
            # Clinical status
            clinical_status = allergy_resource.get('clinicalStatus', {})
            status_codings = clinical_status.get('coding', [])
            if status_codings:
                allergy['status'] = status_codings[0].get('code', 'unknown')
            
            # Type
            allergy['type'] = allergy_resource.get('type', 'unknown')
            
            # Category
            categories = allergy_resource.get('category', [])
            if categories:
                allergy['category'] = categories[0]
            
            # Criticality
            allergy['criticality'] = allergy_resource.get('criticality', 'unknown')
            
            # Reactions
            reactions = allergy_resource.get('reaction', [])
            if reactions:
                reaction_data = []
                for reaction in reactions:
                    manifestations = reaction.get('manifestation', [])
                    for manifestation in manifestations:
                        manifest_codings = manifestation.get('coding', [])
                        if manifest_codings:
                            reaction_data.append(manifest_codings[0].get('display', ''))
                allergy['reactions'] = reaction_data
            
            return allergy
            
        except Exception as e:
            print(f"Warning: Could not parse allergy: {e}")
            return None
    
    def _parse_procedure(self, procedure_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse FHIR Procedure resource"""
        try:
            procedure = {}
            
            # Procedure ID
            procedure['id'] = procedure_resource.get('id', '')
            
            # Procedure code
            code = procedure_resource.get('code', {})
            codings = code.get('coding', [])
            if codings:
                primary_coding = codings[0]
                procedure['code'] = primary_coding.get('code', '')
                procedure['system'] = primary_coding.get('system', '')
                procedure['name'] = primary_coding.get('display', '')
            
            procedure['text'] = code.get('text', procedure.get('name', ''))
            
            # Status
            procedure['status'] = procedure_resource.get('status', 'unknown')
            
            # Performed date
            performed_datetime = procedure_resource.get('performedDateTime')
            if performed_datetime:
                procedure['performed_date'] = performed_datetime
            
            # Outcome
            outcome = procedure_resource.get('outcome', {})
            if outcome:
                outcome_text = outcome.get('text', '')
                procedure['outcome'] = outcome_text
            
            return procedure
            
        except Exception as e:
            print(f"Warning: Could not parse procedure: {e}")
            return None
    
    def _parse_encounter(self, encounter_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse FHIR Encounter resource"""
        try:
            encounter = {}
            
            # Encounter ID
            encounter['id'] = encounter_resource.get('id', '')
            
            # Status
            encounter['status'] = encounter_resource.get('status', 'unknown')
            
            # Class
            class_code = encounter_resource.get('class', {})
            encounter['class'] = class_code.get('display', class_code.get('code', 'unknown'))
            
            # Type
            types = encounter_resource.get('type', [])
            if types:
                type_codings = types[0].get('coding', [])
                if type_codings:
                    encounter['type'] = type_codings[0].get('display', 'unknown')
            
            # Period
            period = encounter_resource.get('period', {})
            encounter['start_date'] = period.get('start')
            encounter['end_date'] = period.get('end')
            
            # Reason
            reason_codes = encounter_resource.get('reasonCode', [])
            if reason_codes:
                reason_codings = reason_codes[0].get('coding', [])
                if reason_codings:
                    encounter['reason'] = reason_codings[0].get('display', '')
            
            return encounter
            
        except Exception as e:
            print(f"Warning: Could not parse encounter: {e}")
            return None
    
    def _calculate_age(self, birth_date: str) -> int:
        """Calculate age from birth date string"""
        try:
            birth = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
            today = datetime.now()
            age = today.year - birth.year
            if today.month < birth.month or (today.month == birth.month and today.day < birth.day):
                age -= 1
            return max(0, age)
        except Exception:
            return 0
    
    def _calculate_duration(self, start_date: str) -> str:
        """Calculate duration from start date to now"""
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            now = datetime.now()
            delta = now - start
            
            years = delta.days // 365
            months = (delta.days % 365) // 30
            days = delta.days % 30
            
            if years > 0:
                return f"{years} year{'s' if years > 1 else ''}"
            elif months > 0:
                return f"{months} month{'s' if months > 1 else ''}"
            else:
                return f"{days} day{'s' if days > 1 else ''}"
                
        except Exception:
            return "unknown duration"
    
    def convert_to_clinical_format(self, parsed_data: ParsedPatientData) -> Dict[str, Any]:
        """
        Convert parsed FHIR data to clinical decision engine format
        
        Args:
            parsed_data: ParsedPatientData from FHIR parsing
            
        Returns:
            Dict formatted for clinical decision engine
        """
        clinical_format = {
            "patient_id": parsed_data.patient_id,
            "demographics": parsed_data.demographics,
            "past_medical_history": [],
            "medications": [],
            "vital_signs": {},
            "laboratory_results": {},
            "allergies": [],
            "procedures": [],
            "chief_complaint": "",  # To be filled by user input
        }
        
        # Convert conditions to medical history
        for condition in parsed_data.conditions:
            if condition.get('status') in ['active', 'confirmed']:
                history_item = condition.get('text', condition.get('display', ''))
                if condition.get('onset_date'):
                    history_item += f" (since {condition['onset_date'][:10]})"
                clinical_format["past_medical_history"].append(history_item)
        
        # Convert medications
        for medication in parsed_data.medications:
            if medication.get('status') in ['active', 'completed']:
                med_text = medication.get('text', medication.get('name', ''))
                if medication.get('dosage_text'):
                    med_text += f" - {medication['dosage_text']}"
                clinical_format["medications"].append(med_text)
        
        # Convert observations to vitals and labs
        for observation in parsed_data.observations:
            obs_name = observation.get('name', observation.get('text', ''))
            obs_value = observation.get('value', '')
            obs_unit = observation.get('unit', '')
            
            # Categorize observations
            category = observation.get('category', '').lower()
            
            if 'vital' in category or any(vital in obs_name.lower() for vital in 
                ['blood pressure', 'heart rate', 'temperature', 'respiratory rate', 'oxygen']):
                clinical_format["vital_signs"][obs_name] = f"{obs_value} {obs_unit}".strip()
            else:
                clinical_format["laboratory_results"][obs_name] = f"{obs_value} {obs_unit}".strip()
        
        # Convert allergies
        for allergy in parsed_data.allergies:
            if allergy.get('status') in ['active', 'confirmed']:
                allergy_text = allergy.get('substance', '')
                if allergy.get('reactions'):
                    allergy_text += f" (reactions: {', '.join(allergy['reactions'])})"
                clinical_format["allergies"].append(allergy_text)
        
        # Add procedures
        for procedure in parsed_data.procedures:
            proc_text = procedure.get('text', procedure.get('name', ''))
            if procedure.get('performed_date'):
                proc_text += f" ({procedure['performed_date'][:10]})"
            clinical_format["procedures"].append(proc_text)
        
        return clinical_format


def create_fhir_parser() -> FHIRDataParser:
    """Factory function to create a FHIR parser instance"""
    return FHIRDataParser()


# Sample FHIR data for testing
SAMPLE_FHIR_BUNDLE = {
    "resourceType": "Bundle",
    "id": "patient-example-bundle",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "patient-001",
                "name": [
                    {
                        "given": ["John"],
                        "family": "Doe"
                    }
                ],
                "gender": "male",
                "birthDate": "1960-05-15",
                "telecom": [
                    {
                        "system": "phone",
                        "value": "555-123-4567"
                    }
                ]
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "condition-hypertension",
                "clinicalStatus": {
                    "coding": [
                        {
                            "code": "active"
                        }
                    ]
                },
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "38341003",
                            "display": "Hypertension"
                        }
                    ]
                },
                "onsetDateTime": "2020-01-15"
            }
        }
    ]
}


if __name__ == "__main__":
    # Example usage
    parser = FHIRDataParser()
    
    # Parse sample FHIR bundle
    parsed_data = parser.parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    
    # Convert to clinical format
    clinical_data = parser.convert_to_clinical_format(parsed_data)
    
    print("\n" + "="*50)
    print("FHIR PARSING RESULT")
    print("="*50)
    print(json.dumps(clinical_data, indent=2, default=str))