import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union



logger = logging.getLogger(__name__)

class VariantEncoder:
    """
    Handles encoding of variant data for SENN model input.
    Creates consistent feature templates and encodes both single and batch inputs.
    """
    
    def __init__(self):
        # Predefined categories : these are the only values for each category it will accept. Extracted from all clinvar data
        self.MC_categories = ['nonsense', 'non-coding_transcript_variant', 'missense_variant',
                             'intron_variant', '5_prime_UTR_variant', 'splice_donor_variant',
                             'synonymous_variant', 'splice_acceptor_variant',
                             'initiator_codon_variant', '3_prime_UTR_variant',
                             'no_sequence_alteration', 'stop_lost',
                             'genic_upstream_transcript_variant',
                             'genic_downstream_transcript_variant']
        
        self.origin_categories = ['germline', 'biparental', 'unknown', 'maternal', 'paternal',
                                 'inherited', 'de novo', 'not applicable', 'tested-inconclusive',
                                 'uniparental', 'not-reported']
        
        self.vgr_categories = ['within single gene', 'within multiple genes by overlap',
                              'asserted, but not computed', 'near gene, upstream',
                              'near gene, downstream', 'not identified']
        
        self.alleles = ['A', 'T', 'G', 'C']
        
        self.chromosomes = ['11', '6', '2', '20', '10', '16', '22', '15', '1', '7', '8', '14',
                           '21', '5', '4', '19', '3', '17', '12', '18', '9', '13', 'MT', 'Y', 'X']
        
        self.genomic_location_categories = ['g', 'm']
        
        # Create feature template
        self.feature_template = self._create_feature_template()

    def _create_feature_template(self) -> pd.DataFrame:
        """Create a template DataFrame with all possible encoded features"""
        template_data = {}
        
        # Multi-hot encoding for MC
        for mc in self.MC_categories:
            template_data[f'has_MC_{mc}'] = 0
            
        # Multi-hot encoding for Origin
        for origin in self.origin_categories:
            template_data[f'has_Origin_{origin}'] = 0
            
        # One-hot encoding for VariantGeneRelation
        for vgr in self.vgr_categories:
            template_data[f'has_VariantGeneRelation_{vgr}'] = 0
            
        # One-hot encoding for Reference and Alternate alleles
        for allele in self.alleles:
            template_data[f'ref_is_{allele}'] = 0
            template_data[f'alt_is_{allele}'] = 0
            
        # One-hot encoding for Chromosomes
        for chrom in self.chromosomes:
            template_data[f'chr_{chrom}'] = 0
            
        # Add genomic location features (encoded from GenomicLocationData)
        template_data['is_genomic'] = 0
        template_data['is_mitochondrial'] = 0
        
        return pd.DataFrame([template_data])

    def _encode_genomic_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode GenomicLocationData into is_genomic and is_mitochondrial features"""
        encoded_df = df.copy()
        
        # Single-hot encoding: 'g' -> is_genomic=1, is_mitochondrial=0
        #                      'm' -> is_genomic=0, is_mitochondrial=1
        encoded_df['is_genomic'] = (df['GenomicLocationData'] == 'g').astype(int)
        encoded_df['is_mitochondrial'] = (df['GenomicLocationData'] == 'm').astype(int)
        
        return encoded_df
    
    def _encode_MC(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-hot encode MC column"""
        encoded_df = df.copy()
        
        for mc in self.MC_categories:
            encoded_df[f'has_MC_{mc}'] = df['MC'].str.contains(mc, na=False).astype(int)
            
        return encoded_df
    
    def _encode_Origin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-hot encode Origin column"""
        encoded_df = df.copy()
        
        for origin in self.origin_categories:
            encoded_df[f'has_Origin_{origin}'] = df['Origin'].str.contains(origin, na=False).astype(int)
            
        return encoded_df
    
    def _encode_VariantGeneRelation(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode VariantGeneRelation column"""
        encoded_df = df.copy()
        
        for vgr in self.vgr_categories:
            encoded_df[f'has_VariantGeneRelation_{vgr}'] = (df['VariantGeneRelation'] == vgr).astype(int)
            
        return encoded_df
    
    def _encode_alleles(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode Reference and Alternate alleles"""
        encoded_df = df.copy()
        
        for allele in self.alleles:
            encoded_df[f'ref_is_{allele}'] = (df['ReferenceAlleleVCF'] == allele).astype(int)
            encoded_df[f'alt_is_{allele}'] = (df['AlternateAlleleVCF'] == allele).astype(int)
            
        return encoded_df
    
    def _encode_chromosome(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode Chromosome"""
        encoded_df = df.copy()
        
        for chrom in self.chromosomes:
            encoded_df[f'chr_{chrom}'] = (df['Chromosome'] == chrom).astype(int)
            
        return encoded_df

    def _log_unknown_categories(self, df: pd.DataFrame) -> List[str]:
        """Log unknown categories and return AlleleIDs with unknown data"""
        unknown_alleles = []
        
        # Check MC categories
        if 'MC' in df.columns:
            all_mc_variants = df['MC'].dropna().str.split(',').explode().unique()
            unknown_mc = set(all_mc_variants) - set(self.MC_categories)
            if unknown_mc:
                affected_ids = df[df['MC'].str.contains('|'.join(unknown_mc), na=False)]['AlleleID'].tolist()
                logger.warning(f"Unknown MC variants found: {unknown_mc} in AlleleIDs: {affected_ids}")
                unknown_alleles.extend(affected_ids)
        
        # Check Origin categories
        if 'Origin' in df.columns:
            all_origins = df['Origin'].dropna().str.split(';').explode().unique()
            unknown_origins = set(all_origins) - set(self.origin_categories)
            if unknown_origins:
                affected_ids = df[df['Origin'].str.contains('|'.join(unknown_origins), na=False)]['AlleleID'].tolist()
                logger.warning(f"Unknown Origin types found: {unknown_origins} in AlleleIDs: {affected_ids}")
                unknown_alleles.extend(affected_ids)
        
        # Check other categorical columns
        categorical_checks = [
            ('VariantGeneRelation', self.vgr_categories),
            ('ReferenceAlleleVCF', self.alleles),
            ('AlternateAlleleVCF', self.alleles),
            ('Chromosome', self.chromosomes),
            ('GenomicLocationData', self.genomic_location_categories)
        ]
        
        for col, valid_categories in categorical_checks:
            if col in df.columns:
                unknown_vals = set(df[col].dropna().unique()) - set(valid_categories)
                if unknown_vals:
                    affected_ids = df[df[col].isin(unknown_vals)]['AlleleID'].tolist()
                    logger.warning(f"Unknown {col} values found: {unknown_vals} in AlleleIDs: {affected_ids}")
                    unknown_alleles.extend(affected_ids)
        
        return list(set(unknown_alleles))

    def encode_single_input(self, input_data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Encode a single input dictionary into model-ready features
        
        Args:
            input_data: Dictionary with keys matching column names
            
        Returns:
            encoded_features: DataFrame with encoded features
            issues: List of issue messages
        """
        issues = []
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Check for missing values in original data
        required_cols = ['AlleleID', 'Origin', 'Chromosome', 'ReferenceAlleleVCF', 
                        'AlternateAlleleVCF', 'VariantGeneRelation', 'MC', 'GenomicLocationData']
        
        missing_data_cols = [col for col in required_cols if col not in df.columns or df[col].isna().any()]
        if missing_data_cols:
            issues.append(f"Missing data in columns: {missing_data_cols}")
            return None, issues
        
        # Log unknown categories
        unknown_alleles = self._log_unknown_categories(df)
        if unknown_alleles:
            issues.extend([f"Unknown categories in AlleleID: {aid}" for aid in unknown_alleles])
        
        # Apply encodings
        df = self._encode_MC(df)
        df = self._encode_Origin(df)
        df = self._encode_VariantGeneRelation(df)
        df = self._encode_alleles(df)
        df = self._encode_chromosome(df)
        df = self._encode_genomic_location(df)
        
        # Select and order features according to template
        feature_cols = [col for col in self.feature_template.columns]
        encoded_features = df[['AlleleID', 'GeneID'] + feature_cols].copy() if 'GeneID' in df.columns else df[['AlleleID'] + feature_cols].copy()
        
        return encoded_features, issues
    
    def encode_batch_input(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Encode batch input DataFrame into model-ready features
        
        Args:
            df: Input DataFrame with required columns
            
        Returns:
            encoded_features: DataFrame with encoded features
            issues: List of AlleleIDs with issues
        """
        issues = []
        
        # Check required columns
        required_cols = ['AlleleID', 'Origin', 'Chromosome', 'ReferenceAlleleVCF', 
                        'AlternateAlleleVCF', 'VariantGeneRelation', 'MC', 'GenomicLocationData']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return None, issues
        
        # Check for rows with missing data
        missing_data_mask = df[required_cols].isnull().any(axis=1)
        missing_allele_ids = df[missing_data_mask]['AlleleID'].tolist()
        
        if missing_allele_ids:
            logger.warning(f"Found {len(missing_allele_ids)} rows with missing data")
            issues.extend(missing_allele_ids)
            # Remove rows with missing data
            df = df[~missing_data_mask].copy()
        
        if len(df) == 0:
            issues.append("No valid rows remaining after removing missing data")
            return None, issues
        
        # Log unknown categories
        unknown_alleles = self._log_unknown_categories(df)
        
        # Apply encodings
        df = self._encode_MC(df)
        df = self._encode_Origin(df)
        df = self._encode_VariantGeneRelation(df)
        df = self._encode_alleles(df)
        df = self._encode_chromosome(df)
        df = self._encode_genomic_location(df)
        
        # Select and order features according to template
        feature_cols = [col for col in self.feature_template.columns]
        id_cols = ['AlleleID'] + (['GeneID'] if 'GeneID' in df.columns else [])
        encoded_features = df[id_cols + feature_cols].copy()
        
        return encoded_features, issues

class VariantEncoderEndpoint:
    """
    Endpoint wrapper for VariantEncoder
    """
    
    def __init__(self):
        self.DB_PATH_SQL = r'C:\Users\vigne\Desktop\Capstone\datasets\Capstone_data_sql.duckdb'
        
        # Initialize the encoder
        self.encoder = VariantEncoder()

    def _read_file(self, file_path: str) -> pd.DataFrame:
        """Read TSV or CSV file and return DataFrame"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and read accordingly
            if file_path.suffix.lower() in ['.tsv', '.txt']:
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                # Try to auto-detect
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    df = pd.read_csv(file_path)
            
            logger.info(f"Successfully read {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def _get_clinical_significance(self, allele_id: int) -> Optional[str]:
        """Get current clinical significance from database"""
        try:
            import duckdb
            conn = duckdb.connect(self.DB_PATH_SQL)
            query = "SELECT ClinicalSignificance FROM allele WHERE AlleleID = ?"
            result = conn.execute(query, [allele_id]).fetchone()
            conn.close()
            
            if result:
                return result[0]
            else:
                logger.warning(f"AlleleID {allele_id} not found in ClinicalSignificance database")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get clinical significance for {allele_id}: {e}")
            return None

    def encode_variant_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Endpoint function for single variant encoding
        
        Args:
            input_data: Dictionary with variant data
            Required: MC, Origin, ReferenceAlleleVCF, AlternateAlleleVCF, Chromosome, VariantGeneRelation, GenomicLocationData
            Optional: AlleleID, GeneID
            
        Returns:
            Dictionary containing:
            - allele_id: Input AlleleID (if provided)
            - gene_id: Input GeneID (if provided) 
            - clinical_significance: Current ClinicalSignificance from DB
            - encoded_features: 66-length array ready for model
            - validation_issues: List of issues/warnings
        """
        
        validation_issues = []
        result = {
            'allele_id': input_data.get('AlleleID'),
            'gene_id': input_data.get('GeneID'),
            'clinical_significance': None,
            'encoded_features': None,
            'validation_issues': []
        }
        
        # Check required fields
        required_fields = ['MC', 'Origin', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Chromosome', 'VariantGeneRelation', 'GenomicLocationData']
        missing_fields = [field for field in required_fields if field not in input_data or not input_data[field]]
        
        if missing_fields:
            validation_issues.append(f"Missing required fields: {missing_fields}")
            result['validation_issues'] = validation_issues
            return result
        
        # Get clinical significance if AlleleID is provided
        if input_data.get('AlleleID'):
            clinical_sig = self._get_clinical_significance(input_data['AlleleID'])
            result['clinical_significance'] = clinical_sig
            if clinical_sig is None:
                validation_issues.append(f"Clinical significance not found for AlleleID: {input_data['AlleleID']}")
        
        # Ensure AlleleID is present for encoding (required by original encoder)
        if 'AlleleID' not in input_data:
            input_data['AlleleID'] = 'temp_id_for_encoding'
        
        # Encode the variant
        try:
            encoded_df, encoder_issues = self.encoder.encode_single_input(input_data)
            
            if encoded_df is None:
                validation_issues.extend(encoder_issues)
                result['validation_issues'] = validation_issues
                return result
            
            # Extract the 66 features (excluding AlleleID, GeneID columns)
            feature_columns = [col for col in encoded_df.columns 
                             if col not in ['AlleleID', 'GeneID']]
            
            if len(feature_columns) != 66:
                validation_issues.append(f"Expected 66 features, got {len(feature_columns)}")
                result['validation_issues'] = validation_issues
                return result
            
            # Get the feature array
            encoded_features = encoded_df[feature_columns].iloc[0].values.astype(np.float32)
            result['encoded_features'] = encoded_features.tolist()
            
            # Add any encoder issues
            if encoder_issues:
                validation_issues.extend(encoder_issues)
            
        except Exception as e:
            validation_issues.append(f"Encoding failed: {str(e)}")
            logger.error(f"Variant encoding failed: {e}")
        
        result['validation_issues'] = validation_issues
        return result

        def encode_variant_batch(self, file_path: str) -> Dict[str, Any]:
        """
        Endpoint function for batch variant encoding from file
        
        Args:
            file_path: Path to TSV or CSV file with variant data
            Required columns: MC, Origin, ReferenceAlleleVCF, AlternateAlleleVCF, Chromosome, VariantGeneRelation, GenomicLocationData
            Optional columns: AlleleID, GeneID
            
        Returns:
            Dictionary containing:
            - total_variants: Total number of variants processed
            - successful_encodings: Number of successfully encoded variants
            - failed_encodings: Number of failed encodings
            - results: List of dictionaries, each containing:
                - allele_id, gene_id, clinical_significance, encoded_features, validation_issues
            - global_issues: File-level issues
        """
        
        global_issues = []
        results = []
        
        try:
            # Read the file
            df = self._read_file(file_path)
            
            if df.empty:
                return {
                    'total_variants': 0,
                    'successful_encodings': 0,
                    'failed_encodings': 0,
                    'results': [],
                    'global_issues': ['File is empty']
                }
            
            # Check required columns
            required_cols = ['MC', 'Origin', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Chromosome', 'VariantGeneRelation', 'GenomicLocationData']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    'total_variants': len(df),
                    'successful_encodings': 0,
                    'failed_encodings': len(df),
                    'results': [],
                    'global_issues': [f"Missing required columns: {missing_cols}"]
                }
            
            # Add temporary AlleleID for rows without it
            for idx, row in df.iterrows():
                if pd.isna(row.get('AlleleID')):
                    df.at[idx, 'AlleleID'] = f'temp_id_{idx}'
            
            # Process using batch encoder
            try:
                encoded_df, batch_issues = self.encoder.encode_batch_input(df)
                
                if encoded_df is None:
                    return {
                        'total_variants': len(df),
                        'successful_encodings': 0,
                        'failed_encodings': len(df),
                        'results': [],
                        'global_issues': batch_issues
                    }
                
                # Extract feature columns (66 features)
                feature_columns = [col for col in encoded_df.columns 
                                 if col not in ['AlleleID', 'GeneID']]
                
                if len(feature_columns) != 66:
                    global_issues.append(f"Expected 66 features, got {len(feature_columns)}")
                
                # Process each row
                successful = 0
                failed = 0
                
                for idx, encoded_row in encoded_df.iterrows():
                    original_row = df.iloc[idx]
                    
                    try:
                        # Get clinical significance if real AlleleID
                        clinical_sig = None
                        allele_id = original_row.get('AlleleID')
                        if allele_id and not str(allele_id).startswith('temp_id_'):
                            clinical_sig = self._get_clinical_significance(str(allele_id))
                        
                        # Extract features
                        encoded_features = encoded_row[feature_columns].values.astype(np.float32)
                        
                        # Create result
                        variant_result = {
                            'allele_id': allele_id if not str(allele_id).startswith('temp_id_') else None,
                            'gene_id': original_row.get('GeneID'),
                            'clinical_significance': clinical_sig,
                            'encoded_features': encoded_features.tolist(),
                            'validation_issues': []
                        }
                        
                        # Add any specific issues for this variant
                        if str(allele_id) in batch_issues:
                            variant_result['validation_issues'].append(f"Batch processing issue: {allele_id}")
                        
                        results.append(variant_result)
                        successful += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process variant at index {idx}: {e}")
                        results.append({
                            'allele_id': allele_id if not str(allele_id).startswith('temp_id_') else None,
                            'gene_id': original_row.get('GeneID'),
                            'clinical_significance': None,
                            'encoded_features': None,
                            'validation_issues': [f"Processing failed: {str(e)}"]
                        })
                        failed += 1
                
                return {
                    'total_variants': len(df),
                    'successful_encodings': successful,
                    'failed_encodings': failed,
                    'results': results,
                    'global_issues': global_issues + batch_issues
                }
                
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                return {
                    'total_variants': len(df),
                    'successful_encodings': 0,
                    'failed_encodings': len(df),
                    'results': [],
                    'global_issues': [f"Batch encoding failed: {str(e)}"]
                }
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {
                'total_variants': 0,
                'successful_encodings': 0,
                'failed_encodings': 0,
                'results': [],
                'global_issues': [f"File processing failed: {str(e)}"]
            }

# Instantiate the endpoint
variant_encoder_endpoint = VariantEncoderEndpoint()

def encode_variant_endpoint(
    input_data: Optional[Dict[str, Any]] = None, 
    file_path: Optional[str] = None, 
    input_type: str = "auto"
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Unified endpoint function for variant encoding - supports both single and batch processing
    
    Args:
        input_data: Dictionary with variant data (for single variant)
        file_path: Path to TSV/CSV file (for batch processing)
        input_type: "single", "batch", or "auto" (auto-detect based on provided args)
        
    Returns:
        For single variant: Dictionary with encoded features and metadata
        For batch: Dictionary with batch processing results and list of variant results
        
    Usage Examples:
        # Single variant
        result = encode_variant_endpoint(
            input_data={
                'AlleleID': '12345',
                'MC': 'missense_variant',
                'Origin': 'germline',
                'ReferenceAlleleVCF': 'A',
                'AlternateAlleleVCF': 'T', 
                'Chromosome': '1',
                'VariantGeneRelation': 'within single gene',
                'GenomicLocationData': 'g'
            },
            input_type="single"
        )
        
        # Batch processing
        result = encode_variant_endpoint(
            file_path="/path/to/variants.tsv",
            input_type="batch"
        )
        
        # Auto-detect (recommended)
        result = encode_variant_endpoint(input_data={...})  # Single
        result = encode_variant_endpoint(file_path="file.tsv")  # Batch
    """
    
    # Auto-detect input type if not specified
    if input_type == "auto":
        if input_data is not None and file_path is None:
            input_type = "single"
        elif file_path is not None and input_data is None:
            input_type = "batch"
        else:
            raise ValueError("Invalid arguments: provide either input_data or file_path")
    
    # Dispatch based on input type
    if input_type == "single":
        return variant_encoder_endpoint.encode_variant_single(input_data)
    elif input_type == "batch":
        return variant_encoder_endpoint.encode_variant_batch(file_path)
    else:
        raise ValueError(f"Invalid input_type: {input_type}. Must be 'single', 'batch', or 'auto'.")



result = encode_variant_endpoint(
    input_data={
        'AlleleID': 15044,
        'GeneID': 55572,
        'Origin': 'germline',
        'Chromosome': '11o',
        'ReferenceAlleleVCF': 'B',
        'AlternateAlleleVCF': 'T',
        'VariantGeneRelation': 'within single genes',
        'MC': 'nonsense,non-coding_transcript_variant',
        'GenomicLocationData': 'g'
    },
    input_type="single"
)
# output schema 
# result = {
#     "allele_id": str | int | None,        # AlleleID if given, else None
#     "gene_id": str | int | None,          # GeneID if given, else None
#     "clinical_significance": str | None,  # Value pulled from DB if found
#     "encoded_features": List[float] | None,  # 66-length array of floats if encoding succeeds
#     "validation_issues": List[str]        # Any warnings/errors/unknown categories
# }

batch_result = encode_variant_endpoint(
    file_path=r"C:\Users\vigne\Desktop\Capstone\datasets\batchtest",
    input_type="batch"
)
# Output schema 
# batch_result = {
#     "total_variants": int,               # Total rows read from file
#     "successful_encodings": int,         # Number successfully encoded
#     "failed_encodings": int,             # Number failed
#     "results": [                         # One dict per variant, same schema as `result`
#         {
#             "allele_id": str | int | None,
#             "gene_id": str | int | None,
#             "clinical_significance": str | None,
#             "encoded_features": List[float] | None,  # 66 features
#             "validation_issues": List[str]
#         },
#         ...
#     ],
#     "global_issues": List[str]           # File-level issues (e.g. missing columns)
# }



print("\n--- Single Variant Result ---")
print("Allele ID:", result["allele_id"])
print("Gene ID:", result["gene_id"])
print("Clinical Significance:", result["clinical_significance"])
print("Validation Issues:", result["validation_issues"])
print("Encoded Features Length:", len(result["encoded_features"]) if result["encoded_features"] else 0)
print("First 10 Encoded Features:", result["encoded_features"][:10] if result["encoded_features"] else None)



print("\n--- Batch Variant Result ---")
print("Total Variants:", batch_result["total_variants"])
print("Successful Encodings:", batch_result["successful_encodings"])
print("Failed Encodings:", batch_result["failed_encodings"])
print("Global Issues:", batch_result["global_issues"])

print("\n--- First Variant Result in Batch ---")
if batch_result["results"]:
    first_var = batch_result["results"][0]
    print("Allele ID:", first_var["allele_id"])
    print("Gene ID:", first_var["gene_id"])
    print("Clinical Significance:", first_var["clinical_significance"])
    print("Validation Issues:", first_var["validation_issues"])
    print("Encoded Features Length:", len(first_var["encoded_features"]) if first_var["encoded_features"] else 0)
    print("First 10 Encoded Features:", first_var["encoded_features"][:10] if first_var["encoded_features"] else None)