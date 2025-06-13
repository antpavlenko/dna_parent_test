import os
import click

# Handle both relative imports (when run as module) and absolute imports (when run directly)
try:
    from .s3_utils import kinship_from_s3_files, test_s3_connection
except ImportError:
    # If relative imports fail, try absolute imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dna_parent_test.s3_utils import kinship_from_s3_files, test_s3_connection


@click.command()
@click.option('--subject1', help='S3 path to first subject CRAM file (s3://bucket/subject1.cram)')
@click.option('--subject2', help='S3 path to second subject CRAM file (s3://bucket/subject2.cram)')
@click.option('--reference', help='Reference genome path (local file)')
@click.option('--region', default='chr22', help='Genomic region (default: chr22). Examples: chr22, chr21,chr22, chr1:1000000-2000000')
@click.option('--s3-endpoint', required=True, help='S3 endpoint URL (required, e.g., https://ru.serverspace.store:443/)')
@click.option('--s3-access-key', required=True, help='S3 access key')
@click.option('--s3-secret-key', required=True, help='S3 secret key')
@click.option('--s3-region', default='us-east-1', help='S3 region (default: us-east-1)')
@click.option('--test-s3', help='Test S3 connection with bucket name')

@click.option('--output-prefix', default='kinship_analysis', help='Output file prefix')
def main(subject1: str, subject2: str, reference: str, region: str, s3_endpoint: str,
         s3_access_key: str, s3_secret_key: str, s3_region: str, test_s3: str,
         output_prefix: str):
    """
    DNA Kinship Analysis from CRAM files in S3 storage.

    Performs streaming variant calling and kinship analysis without local downloads.
    Designed for unbiased, accurate relationship determination.

    Examples:

    \b
    # Basic CRAM analysis (serverspace.kz)
    python cli.py --father s3://bucket/father.cram --child s3://bucket/child.cram \\
                  --reference /path/to/reference.fa \\
                  --s3-endpoint https://ru.serverspace.store:443/ \\
                  --s3-access-key sskz7640_admin \\
                  --s3-secret-key sC7ofL0FQM8n

    \b
    # Analyze specific region (faster)
    python cli.py --father s3://bucket/father.cram --child s3://bucket/child.cram \\
                  --reference /path/to/reference.fa \\
                  --region chr1:1000000-10000000 \\
                  --s3-endpoint https://ru.serverspace.store:443/ \\
                  --s3-access-key sskz7640_admin \\
                  --s3-secret-key sC7ofL0FQM8n

    \b
    # Include STR analysis
    python cli.py --father s3://bucket/father.cram --child s3://bucket/child.cram \\
                  --reference /path/to/reference.fa \\
                  --s3-endpoint https://ru.serverspace.store:443/ \\
                  --s3-access-key sskz7640_admin \\
                  --s3-secret-key sC7ofL0FQM8n \\


    \b
    # Test S3 connection
    python cli.py --test-s3 mybucket \\
                  --s3-endpoint https://ru.serverspace.store:443/ \\
                  --s3-access-key sskz7640_admin \\
                  --s3-secret-key sC7ofL0FQM8n
    """

    # Test S3 connection if requested
    if test_s3:
        success = test_s3_connection(s3_endpoint, s3_access_key, s3_secret_key, test_s3, s3_region)
        if success:
            click.echo("✅ S3 connection test successful!")
        else:
            click.echo("❌ S3 connection test failed!", err=True)
        return

    # Validate inputs for kinship analysis
    if not test_s3:
        if not subject1 or not subject2:
            click.echo("Error: Both --subject1 and --subject2 are required for kinship analysis", err=True)
            return

        if not reference:
            click.echo("Error: --reference is required for CRAM analysis", err=True)
            return
        if not os.path.exists(reference):
            click.echo(f"Error: Reference genome not found: {reference}", err=True)
            return

        if not subject1.startswith('s3://') or not subject2.startswith('s3://'):
            click.echo("Error: Both subject1 and subject2 must be S3 CRAM paths (s3://bucket/file.cram)", err=True)
            return

        if not (subject1.lower().endswith('.cram') and subject2.lower().endswith('.cram')):
            click.echo("Error: Only CRAM files are supported. VCF files are unreliable for kinship analysis.", err=True)
            click.echo("Please use CRAM files for accurate results.", err=True)
            return

    try:
        click.echo("🧬 Starting CRAM streaming analysis from S3 (no local downloads)...")
        click.echo(f"Subject 1: {subject1}")
        click.echo(f"Subject 2: {subject2}")
        click.echo(f"Reference: {reference}")
        if region:
            click.echo(f"Region: {region}")

        kin, ibs0 = kinship_from_s3_files(
            subject1, subject2, s3_endpoint, s3_access_key, s3_secret_key,
            s3_region, reference, output_prefix, region
        )

        click.echo(f'\n📊 Results:')
        click.echo(f'Kinship={kin:.3f} IBS0={ibs0:.3f}')



    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()
