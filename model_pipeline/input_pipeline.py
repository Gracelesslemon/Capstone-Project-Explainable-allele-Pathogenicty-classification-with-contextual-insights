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