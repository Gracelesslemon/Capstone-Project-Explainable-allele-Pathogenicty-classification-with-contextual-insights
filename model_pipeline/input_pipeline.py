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