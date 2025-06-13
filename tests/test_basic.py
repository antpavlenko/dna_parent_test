import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
from dna_parent_test.vcf_utils import merge_vcfs, load_joint_vcf
from dna_parent_test.kinship import kinship_ibd

FATHER_VCF = "father.vcf"
CHILD_VCF = "child.vcf"

FATHER_CONTENT = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=1>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tFATHER
1\t1000\t.\tA\tG\t.\t.\t.\tGT\t0/1
1\t2000\t.\tT\tC\t.\t.\t.\tGT\t0/0
"""

CHILD_CONTENT = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=1>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tCHILD
1\t1000\t.\tA\tG\t.\t.\t.\tGT\t0/1
1\t2000\t.\tT\tC\t.\t.\t.\tGT\t0/1
"""

def setup_vcf(path, content):
    with open(path, 'w') as fh:
        fh.write(content)
    os.system(f"bgzip -c {path} > {path}.gz")
    os.system(f"tabix -p vcf {path}.gz")


def test_kinship(tmp_path):
    father = tmp_path / FATHER_VCF
    child = tmp_path / CHILD_VCF
    setup_vcf(father, FATHER_CONTENT)
    setup_vcf(child, CHILD_CONTENT)

    merged = tmp_path / 'duo.vcf.gz'
    merge_vcfs(f'{father}.gz', f'{child}.gz', merged)
    gt = load_joint_vcf(merged)
    kin, ibs0 = kinship_ibd(gt)
    assert kin > 0.4
    assert ibs0 == 0.0
