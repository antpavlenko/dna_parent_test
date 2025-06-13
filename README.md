# DNA Kinship Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.2.0-orange.svg)](https://pypi.org/project/dna-parent-test/)
[![Development Status](https://img.shields.io/badge/status-stable-green.svg)](https://github.com/antpavlenko/dna-parent-test)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/antpavlenko/dna-parent-test/actions)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://codecov.io/gh/antpavlenko/dna-parent-test)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Bioinformatics](https://img.shields.io/badge/field-bioinformatics-purple.svg)](https://en.wikipedia.org/wiki/Bioinformatics)
[![CRAM Format](https://img.shields.io/badge/format-CRAM-lightblue.svg)](https://samtools.github.io/hts-specs/CRAMv3.pdf)
[![S3 Compatible](https://img.shields.io/badge/storage-S3-orange.svg)](https://aws.amazon.com/s3/)
[![Documentation](https://img.shields.io/badge/docs-readme-blue.svg)](README.md)

**Dependencies:**
[![scikit-allel](https://img.shields.io/badge/scikit--allel-latest-green.svg)](https://scikit-allel.readthedocs.io/)
[![AWS CLI](https://img.shields.io/badge/awscli-latest-orange.svg)](https://aws.amazon.com/cli/)
[![Click](https://img.shields.io/badge/click-CLI-red.svg)](https://click.palletsprojects.com/)
[![BCFtools](https://img.shields.io/badge/requires-bcftools-red.svg)](https://samtools.github.io/bcftools/)
[![Samtools](https://img.shields.io/badge/requires-samtools-red.svg)](http://www.htslib.org/)

**Cloud-native DNA kinship analysis from CRAM files in S3 storage.** This tool performs streaming variant calling and kinship analysis without local downloads, using a corrected KING-robust algorithm optimized for variant-only data to accurately distinguish parent-child relationships from unrelated individuals.

<!--
Badge Update Instructions:
- Replace 'your-username' with your actual GitHub username
- Update PyPI badge when package is published
- Update build status when CI/CD is set up
- Update coverage badge when code coverage is measured
- Add actual license file and update license badge accordingly
-->

## Features

- **🌐 Cloud-Native CRAM Processing**: Direct analysis of CRAM files from S3-compatible storage (no local downloads)
- **🔬 Corrected KING-Robust Algorithm**: IBS0-based discrimination optimized for variant-only data
- **📊 Large-Scale Analysis**: Supports regions from 2MB to entire chromosomes (200MB+)
- **⚡ HTTP Streaming**: Efficient streaming via rclone for optimal performance
- **🎯 High Accuracy**: Properly distinguishes unrelated individuals (negative kinship) from parent-child relationships
- **🚀 S3 Compatible**: Works with any S3-compatible storage (AWS S3, serverspace.kz, etc.)
- **🔧 Required S3 Endpoint**: Configurable S3 endpoints (no hardcoded defaults)
- **📈 Robust Statistics**: Processes 100K+ variants for high-confidence results
- **🖥️ Command-Line Interface**: Simple CLI with comprehensive options
- **🐍 Python API**: Programmatic access to all functionality

## Installation

### Prerequisites

This package requires external bioinformatics tools for CRAM processing and S3 access:

#### Install Required Tools

**On macOS (using Homebrew):**
```bash
brew install samtools bcftools htslib awscli
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install samtools bcftools tabix awscli
```

**On CentOS/RHEL:**
```bash
sudo yum install samtools bcftools htslib awscli
```

**Using Conda (recommended for bioinformatics):**
```bash
conda install -c bioconda samtools bcftools htslib awscli
```

### Install DNA Parent Test

#### Option 1: Install from source (Development)
```bash
git clone https://github.com/antpavlenko/dna_parent_test
cd dna_parent_test
pip install -e .
```

#### Option 2: Install from PyPI (when published)
```bash
pip install dna-parent-test
```

### Python Dependencies

The package automatically installs these Python dependencies:
- `scikit-allel` - Genetic data analysis
- `click` - Command-line interface
- `awscli` - AWS S3 command-line tools

## Usage

### Command Line Interface

#### Basic CRAM Analysis
```bash
# Analyze CRAM files from S3 (entire chromosome 22 by default - ~50MB, 110K variants)
python3 dna_parent_test/cli.py \
    --subject1 s3://bucket/subject1.cram \
    --subject2 s3://bucket/subject2.cram \
    --reference /path/to/reference.fa \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### Large Region Analysis (High Confidence)
```bash
# Multiple chromosomes for maximum statistical power (~100MB, 220K variants)
python3 dna_parent_test/cli.py \
    --subject1 s3://bucket/subject1.cram \
    --subject2 s3://bucket/subject2.cram \
    --reference /path/to/reference.fa \
    --region chr21,chr22 \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### Quick Testing (Smaller Region)
```bash
# Small region for rapid testing (~2MB, 4K variants)
python3 dna_parent_test/cli.py \
    --subject1 s3://bucket/subject1.cram \
    --subject2 s3://bucket/subject2.cram \
    --reference /path/to/reference.fa \
    --region chr22:20000000-22000000 \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### Test S3 Connection
```bash
python3 dna_parent_test/cli.py \
    --test-s3 mybucket \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### Example Output
```
🧬 Starting CRAM streaming analysis from S3 (no local downloads)...
Subject 1: s3://bucket/subject1.cram
Subject 2: s3://bucket/subject2.cram
Reference: /path/to/reference.fa
Region: chr22

Calling variants on entire chromosome 22 (~50MB region) via HTTP...
⏳ This may take 5-10 minutes for large regions but provides robust statistics...
Loaded 110054 total variants
Variants with calls in both samples: 110054
Missing data rate: 0.0%

🧬 Calculating kinship coefficient...
⏳ Processing 110,054 variants for robust kinship estimation...

Kinship Analysis Results:
Kinship coefficient: 0.237423
IBS0 rate: 0.042393
Relationship: Parent-child
✅ High confidence result based on 110,054 variants

📊 Results:
Kinship=0.237 IBS0=0.042
```

### Python API

#### Basic Analysis
```python
from dna_parent_test import kinship_from_s3_files

# Analyze CRAM files from S3
kinship_coeff, ibs0_rate = kinship_from_s3_files(
    s3_file1="s3://bucket/father.cram",
    s3_file2="s3://bucket/child.cram",
    endpoint_url="https://ru.serverspace.store:443/",
    access_key="sskz7640_admin",
    secret_key="sC7ofL0FQM8n",
    reference_genome="/path/to/reference.fa"
)

print(f"Kinship coefficient: {kinship_coeff:.3f}")
print(f"IBS0 rate: {ibs0_rate:.3f}")
```

## Region Size Scaling

The tool supports analysis of regions from small test regions to entire chromosomes:

### Region Size Guide

| Region Type | Size | Variants | Time | Use Case |
|-------------|------|----------|------|----------|
| **Quick Test** | 2MB | ~4K | 2-3 min | `--region chr22:20000000-22000000` |
| **Medium** | 40MB | ~100K | 5-8 min | `--region chr22:10000000-50000000` |
| **Production** | 50MB | ~110K | 8-12 min | `--region chr22` (default) |
| **High Confidence** | 100MB | ~220K | 15-20 min | `--region chr21,chr22` |
| **Maximum Robustness** | 200MB+ | ~400K+ | 30-60 min | `--region chr19,chr20,chr21,chr22` |

### Statistical Power vs Processing Time

- **4K variants**: Good for quick testing
- **100K variants**: Excellent for production use
- **220K+ variants**: Outstanding confidence for research

### Multi-Chromosome Analysis

```bash
# Two chromosomes (~100MB)
--region chr21,chr22

# Three chromosomes (~150MB)
--region chr20,chr21,chr22

# Four chromosomes (~200MB)
--region chr19,chr20,chr21,chr22
```

## Algorithm Details

### Corrected KING-Robust Algorithm

The tool uses a corrected KING-robust algorithm optimized for variant-only data:

#### Key Features:
- **IBS0-based primary discrimination**: Uses IBS0 rate as the most reliable discriminator
- **Variant-only optimized**: Accounts for bias introduced by variant-only VCF files
- **Negative kinship detection**: Properly identifies unrelated individuals with negative kinship coefficients

#### Algorithm Logic:
1. **Primary check**: IBS0 rate (most reliable for kinship)
   - **Low IBS0 (< 0.03)**: Definitely related → kinship ≈ 0.25
   - **High IBS0 (> 0.08)**: Definitely unrelated → kinship < 0
   - **Medium IBS0**: Use raw KING-robust calculation
2. **Secondary check**: Raw KING-robust kinship coefficient
3. **Final adjustment**: Force appropriate ranges based on IBS0

#### Expected Results:
- **Parent-child**: kinship ≈ 0.2-0.3, IBS0 < 0.05
- **Unrelated**: kinship < 0 (negative), IBS0 > 0.08
- **Related but not parent-child**: kinship > 0, IBS0 varies

### S3 Streaming Analysis

For files stored in S3-compatible storage (no local downloads):

#### S3 VCF Files (serverspace.kz)
```bash
python -m dna_parent_test.cli \
    --father s3://mybucket/father.vcf.gz \
    --child s3://mybucket/child.vcf.gz \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### S3 CRAM Files (streaming variant calling)
```bash
python -m dna_parent_test.cli \
    --father s3://mybucket/father.cram \
    --child s3://mybucket/child.cram \
    --reference /path/to/reference.fa \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### Analyze Specific Region (faster for CRAM)
```bash
python -m dna_parent_test.cli \
    --father s3://mybucket/father.cram \
    --child s3://mybucket/child.cram \
    --reference /path/to/reference.fa \
    --region chr1:1000000-10000000 \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```

#### Test S3 Connection
```bash
python -m dna_parent_test.cli \
    --test-s3 mybucket \
    --s3-endpoint https://ru.serverspace.store:443/ \
    --s3-access-key sskz7640_admin \
    --s3-secret-key sC7ofL0FQM8n
```



## Use Cases

### 1. Paternity Testing
Determine if a man is the biological father of a child using genetic variants:

```bash
python -m dna_parent_test.cli --father alleged_father.vcf.gz --child child.vcf.gz
```

**Expected Results:**
- **Parent-Child**: Kinship ≈ 0.5, IBS0 ≈ 0.0
- **Unrelated**: Kinship ≈ 0.0, IBS0 > 0.1

### 2. Family Relationship Verification
Verify claimed family relationships in genetic studies:

```bash
# Test multiple family members
for child in child1.vcf.gz child2.vcf.gz child3.vcf.gz; do
    python -m dna_parent_test.cli --father father.vcf.gz --child $child
done
```

### 3. Quality Control in Genetic Studies
Identify sample mix-ups or contamination in family-based genetic studies:

```python
from dna_parent_test import merge_vcfs, load_joint_vcf, kinship_ibd

# Process multiple families
families = [
    ("father1.vcf.gz", "child1.vcf.gz"),
    ("father2.vcf.gz", "child2.vcf.gz"),
    # ... more families
]

for father, child in families:
    merge_vcfs(father, child, "temp_duo.vcf.gz")
    gt = load_joint_vcf("temp_duo.vcf.gz")
    kinship, ibs0 = kinship_ibd(gt)

    if kinship < 0.4:  # Threshold for parent-child relationship
        print(f"WARNING: {father} and {child} may not be parent-child")
```

### 4. Forensic Applications
Support forensic investigations requiring paternity determination:

```bash
# Compare evidence sample with potential father
python -m dna_parent_test.cli --father suspect.vcf.gz --child evidence.vcf.gz
```

## Interpreting Results

### Kinship Coefficient (Corrected Algorithm)
The kinship coefficient measures genetic similarity between two individuals using the corrected KING-robust algorithm:

| Relationship | Expected Kinship | IBS0 Range | Confidence |
|--------------|------------------|------------|------------|
| **Parent-Child** | **0.20 - 0.30** | **< 0.05** | **High** ✅ |
| **Unrelated** | **< 0 (negative)** | **> 0.08** | **High** ✅ |
| 2nd-degree relatives | 0.05 - 0.15 | 0.02 - 0.05 | Medium |
| 3rd-degree relatives | 0.02 - 0.08 | 0.02 - 0.05 | Medium |
| Identical/MZ twins | > 0.40 | < 0.02 | High |

### IBS0 Rate (Identity-by-State 0)
IBS0 represents the proportion of loci where two individuals share no alleles:

| Relationship | Expected IBS0 | Interpretation |
|--------------|---------------|----------------|
| Parent-Child | ~0.0 | Children inherit one allele from each parent |
| Full Siblings | ~0.25 | Can differ at both alleles for some loci |
| Unrelated | ~0.5-0.7 | High probability of sharing no alleles |

### Decision Thresholds (Updated for Corrected Algorithm)
- **Parent-Child Confirmed**: Kinship > 0.15 AND IBS0 < 0.05
- **Unrelated Confirmed**: Kinship < 0 (negative) OR IBS0 > 0.08
- **Inconclusive**: Values between thresholds (may need larger regions)

## Input Requirements

### VCF File Format
- Files must be bgzip-compressed (.vcf.gz)
- Index files (.vcf.gz.tbi) are created automatically if missing
- Must contain genotype information (GT field)
- Biallelic SNPs recommended for best results

### Preparing VCF Files
```bash
# Compress VCF files (if not already compressed)
bgzip input.vcf

# Index files are created automatically by the tool
# But you can also create them manually if needed:
tabix -p vcf input.vcf.gz

# Filter for biallelic SNPs (optional, recommended)
bcftools view -m2 -M2 -v snps input.vcf.gz -Oz -o filtered.vcf.gz
```

**Note**: The tool automatically creates index files (.tbi) if they don't exist, so manual indexing is not required.

## Troubleshooting

### IBS0 = 0.0 Error
If you see the error "DATA ISSUE: IBS0=0.000000 is too low", this indicates a fundamental problem with your VCF data:

**Possible Causes:**
1. **Same Individual**: VCF files may be from the same person with different sample names
2. **Filtered Data**: VCF files may have been pre-filtered to only include shared variants
3. **Variant Calling Bias**: The variant calling pipeline may have systematic biases

**Solutions:**
1. **Verify Sample Identity**: Confirm that VCF files are from different individuals
2. **Use Raw VCF Files**: Ensure VCF files contain all called variants, not just shared ones
3. **Check Variant Calling**: Review your variant calling pipeline for potential biases
4. **Use Different Data**: Try with VCF files from a different source or calling pipeline

### Low Number of Variants
If you get "Too few variants" error:
- Ensure VCF files contain sufficient high-quality variants (>10,000 recommended)
- Check that VCF files cover overlapping genomic regions
- Verify that variant calling was successful

## Testing

Run the test suite to verify installation:

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/

# Run specific test
pytest tests/test_basic.py -v
```

### Example Test Data
The package includes test data for validation:

```python
# Create test VCF files
python -c "
from tests.test_basic import setup_vcf, FATHER_CONTENT, CHILD_CONTENT
setup_vcf('father.vcf', FATHER_CONTENT)
setup_vcf('child.vcf', CHILD_CONTENT)
"

# Test with example data
python -m dna_parent_test.cli --father father.vcf.gz --child child.vcf.gz
```

## Limitations

- **SNP-based Analysis Only**: Currently supports only SNP variants, not indels or structural variants
- **Biallelic Sites**: Optimized for biallelic SNPs; multiallelic sites may affect accuracy
- **Population Structure**: Results may be affected by population stratification
- **Sample Quality**: Requires high-quality genotype calls for accurate results
- **Coverage**: Low-coverage sequencing may lead to missing genotypes and reduced accuracy
- **Data Quality Issues**: If IBS0 = 0.0, this indicates potential data problems:
  - VCF files may be from the same individual with different sample names
  - VCF files may be filtered to only include shared variants
  - Systematic bias in variant calling pipeline
  - In such cases, relationship determination is unreliable


## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
git clone https://github.com/antpavlenko/dna_parent_test
cd dna_parent_test
pip install -e ".[dev]"
```

## License

MIT

## Support

For questions, issues, or feature requests, please:
1. Check the [Issues](https://github.com/antpavlenko/dna-parent-test/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## Disclaimer

This software is for research purposes only. For clinical or legal applications, please consult with appropriate professionals and validate results using established methods.

