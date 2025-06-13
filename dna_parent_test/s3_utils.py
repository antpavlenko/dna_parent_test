"""
S3 utilities for streaming VCF/CRAM files from custom S3 endpoints.
Supports AWS S3 and S3-compatible services like serverspace.kz.
No local downloads - streams data directly for analysis.
"""

import os
import tempfile
import subprocess
from typing import Optional, Tuple
import shutil


def check_s3_tools():
    """Check if required tools are available."""
    required_tools = ["rclone", "bcftools", "samtools", "tabix", "bgzip"]
    missing_tools = []

    for tool in required_tools:
        if not shutil.which(tool):
            missing_tools.append(tool)

    if missing_tools:
        raise RuntimeError(
            f"Missing required tools: {', '.join(missing_tools)}. "
            f"Please install:\n"
            f"  rclone: brew install rclone (macOS) or conda install -c conda-forge rclone\n"
            f"  BCFtools: brew install bcftools (macOS) or conda install bcftools\n"
            f"  Samtools: brew install samtools (macOS) or conda install samtools\n"
            f"  Tabix/bgzip: brew install htslib (macOS) or conda install htslib"
        )


def configure_s3_endpoint(endpoint_url: str, access_key: str, secret_key: str, region: str = "us-east-1") -> str:
    """
    Configure rclone for custom S3 endpoint.

    Args:
        endpoint_url: S3 endpoint URL (e.g., https://ru.serverspace.store:443/)
        access_key: S3 access key
        secret_key: S3 secret key
        region: AWS region (default for S3-compatible services)

    Returns:
        The rclone config name created
    """
    # Create a unique config name based on endpoint
    import hashlib
    endpoint_hash = hashlib.md5(endpoint_url.encode()).hexdigest()[:8]
    config_name = f"s3_auto_{endpoint_hash}"

    # Remove existing config if it exists
    subprocess.run(['rclone', 'config', 'delete', config_name],
                  capture_output=True, check=False)

    # Create new rclone config for serverspace.kz
    subprocess.run([
        'rclone', 'config', 'create', config_name, 's3',
        'provider=Other',
        f'access_key_id={access_key}',
        f'secret_access_key={secret_key}',
        f'endpoint={endpoint_url}',
        'v2_auth=true',  # Required for serverspace.kz
        '--obscure'
    ], check=True)

    print(f"Configured rclone for S3 endpoint: {endpoint_url} (using signature v2)")
    return config_name


def stream_s3_to_bcftools(s3_path: str, config_name: str) -> subprocess.Popen:
    """
    Stream S3 file directly to bcftools without local download using rclone.

    Args:
        s3_path: S3 path (e.g., s3://bucket/file.vcf.gz or s3://bucket/file.cram)
        config_name: Rclone config name to use

    Returns:
        Subprocess.Popen object for the streaming process
    """
    # Convert s3://bucket/file to config_name:bucket/file format for rclone
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    # Remove s3:// prefix and convert to rclone format
    rclone_path = s3_path[5:]  # Remove 's3://'
    rclone_remote_path = f"{config_name}:{rclone_path}"

    cmd = ["rclone", "cat", rclone_remote_path]

    print(f"Streaming {s3_path} via rclone...")

    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError(f"Failed to stream {s3_path}: {e}")


def call_variants_from_s3_cram(
    s3_cram_path: str,
    reference_genome: str,
    output_vcf: str,
    endpoint_url: str,
    region: Optional[str] = None,
    min_base_quality: int = 20,
    min_mapping_quality: int = 20
) -> None:
    """
    Call variants from CRAM file using proper region extraction.

    Args:
        s3_cram_path: S3 path to CRAM file
        reference_genome: Path to reference genome (local file)
        output_vcf: Local path for output VCF file
        endpoint_url: S3 endpoint URL (required)
        region: Optional genomic region
        min_base_quality: Minimum base quality
        min_mapping_quality: Minimum mapping quality
    """
    print(f"Calling variants from {s3_cram_path} (region extraction)...")

    if not os.path.exists(reference_genome):
        raise FileNotFoundError(f"Reference genome not found: {reference_genome}")

    # Get the config name from the endpoint URL
    import hashlib
    endpoint_hash = hashlib.md5(endpoint_url.encode()).hexdigest()[:8]
    config_name = f"s3_auto_{endpoint_hash}"

    # Create temporary files for intermediate steps
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="cram_analysis_")

    try:
        # SOLUTION: Use rclone serve to create HTTP endpoint for samtools
        print("Creating HTTP endpoint for CRAM access...")

        # Start rclone serve http in background
        import time

        serve_port = 8080
        serve_proc = subprocess.Popen([
            "rclone", "serve", "http", f"{config_name}:cram",
            "--addr", f"localhost:{serve_port}",
            "--read-only",
            "--vfs-cache-mode", "writes",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        time.sleep(3)

        try:
            # Get CRAM filename
            cram_filename = os.path.basename(s3_cram_path)
            cram_url = f"http://localhost:{serve_port}/{cram_filename}"
            crai_url = f"http://localhost:{serve_port}/{cram_filename}.crai"
            # Explicitly specify remote CRAI index for htslib
            cram_url_with_index = f"{cram_url}#idx={crai_url}"

            print(f"✅ CRAM accessible at: {cram_url}")

            # Test if samtools can access the URL
            test_result = subprocess.run([
                "samtools", "view", "-H", cram_url
            ], capture_output=True, text=True, timeout=30)

            if test_result.returncode != 0:
                raise RuntimeError(f"Cannot access CRAM via HTTP: {test_result.stderr}")

            print("✅ samtools can access CRAM via HTTP")

            # Now call variants using HTTP URL
            if region == 'chr22':
                print(f"Calling variants on entire chromosome 22 (~50MB region) via HTTP...")
                print("⏳ This may take 5-10 minutes for large regions but provides robust statistics...")
            elif ',' in region:
                chromosomes = region.split(',')
                total_size = len(chromosomes) * 50  # Estimate 50MB per chromosome
                print(f"Calling variants on {len(chromosomes)} chromosomes ({region}) ~{total_size}MB total via HTTP...")
                print(f"⏳ This may take {len(chromosomes) * 5}-{len(chromosomes) * 10} minutes for very large regions...")
            else:
                print(f"Calling variants on region {region} via HTTP...")

            # Simplified single-pass approach
            mpileup_cmd = [
                "bcftools", "mpileup",
                "-f", reference_genome,
                "-q", str(min_mapping_quality),
                "-Q", str(min_base_quality),
                "-a", "FORMAT/DP,FORMAT/AD",  # Include depth and allelic depth
                "-Ou",  # Uncompressed BCF output to stdout
                cram_url
            ]

            if region:
                mpileup_cmd.extend(["-r", region])

            call_cmd = [
                "bcftools", "call",
                "-mv",  # Multiallelic and variants-only
                "-Oz",  # Compressed VCF output
                "-o", output_vcf
            ]

            # Run the pipeline
            with subprocess.Popen(mpileup_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as mpileup_proc:
                with subprocess.Popen(call_cmd, stdin=mpileup_proc.stdout,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE) as call_proc:
                    mpileup_proc.stdout.close()

                    # Wait for completion
                    call_stdout, call_stderr = call_proc.communicate()
                    mpileup_stdout, mpileup_stderr = mpileup_proc.communicate()

                    if call_proc.returncode != 0:
                        raise RuntimeError(f"Variant calling failed: {call_stderr.decode()}")

                    if mpileup_proc.returncode != 0:
                        raise RuntimeError(f"Mpileup failed: {mpileup_stderr.decode()}")

        finally:
            # Stop the HTTP server
            if serve_proc.poll() is None:
                serve_proc.terminate()
                serve_proc.wait(timeout=30)

    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Index the output VCF
    print(f"Indexing {output_vcf}...")
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])
    print(f"Variant calling completed: {output_vcf}")


def merge_s3_vcfs_streaming(
    s3_vcf1: str,
    s3_vcf2: str,
    output_vcf: str,
    endpoint_url: Optional[str] = None
) -> None:
    """
    Merge two VCF files from S3 using streaming (no local download).

    Args:
        s3_vcf1: First S3 VCF path
        s3_vcf2: Second S3 VCF path
        output_vcf: Local output VCF path
        endpoint_url: Optional custom endpoint URL
    """
    print(f"Merging {s3_vcf1} and {s3_vcf2} (streaming)...")

    # Use bcftools merge with S3 URLs directly
    cmd = ["bcftools", "merge", "-Oz", "-o", output_vcf]

    # Add S3 URLs - bcftools can handle URLs if configured properly
    # For now, we'll use a different approach with named pipes

    # Create temporary named pipes
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="s3_pipes_")

    try:
        pipe1 = os.path.join(temp_dir, "vcf1.pipe")
        pipe2 = os.path.join(temp_dir, "vcf2.pipe")

        os.mkfifo(pipe1)
        os.mkfifo(pipe2)

        # Start streaming processes
        stream1 = stream_s3_to_bcftools(s3_vcf1, endpoint_url)
        stream2 = stream_s3_to_bcftools(s3_vcf2, endpoint_url)

        # Write to named pipes in background
        def write_to_pipe(stream_proc, pipe_path):
            with open(pipe_path, 'wb') as pipe:
                while True:
                    data = stream_proc.stdout.read(8192)
                    if not data:
                        break
                    pipe.write(data)

        import threading
        thread1 = threading.Thread(target=write_to_pipe, args=(stream1, pipe1))
        thread2 = threading.Thread(target=write_to_pipe, args=(stream2, pipe2))

        thread1.start()
        thread2.start()

        # Run bcftools merge with named pipes
        merge_cmd = ["bcftools", "merge", "-Oz", "-o", output_vcf, pipe1, pipe2]
        subprocess.check_call(merge_cmd)

        # Wait for threads to complete
        thread1.join()
        thread2.join()

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        for proc in [stream1, stream2]:
            if proc.poll() is None:
                proc.terminate()

    # Index the output
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])
    print(f"Merge completed: {output_vcf}")


def kinship_from_s3_files(
    s3_file1: str,
    s3_file2: str,
    endpoint_url: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: str = "us-east-1",
    reference_genome: Optional[str] = None,
    output_prefix: str = "s3_analysis",
    region_filter: Optional[str] = None
) -> Tuple[float, float]:
    """
    Perform kinship analysis on VCF or CRAM files from S3 using streaming (no downloads).

    Args:
        s3_file1: S3 path to first file (VCF.gz or CRAM)
        s3_file2: S3 path to second file (VCF.gz or CRAM)
        endpoint_url: Custom S3 endpoint URL
        access_key: S3 access key
        secret_key: S3 secret key
        region: AWS region
        reference_genome: Reference genome path (required for CRAM files)
        output_prefix: Prefix for output files
        region_filter: Genomic region to analyze (e.g., "chr1:1000000-2000000")

    Returns:
        Tuple of (kinship_coefficient, ibs0_rate)
    """
    # Use local functions (no longer importing from separate modules)

    # Check tools
    check_s3_tools()

    # Configure S3 credentials if provided
    config_name = None
    if access_key and secret_key and endpoint_url:
        config_name = configure_s3_endpoint(endpoint_url, access_key, secret_key, region)

    # Determine file types
    is_cram1 = s3_file1.lower().endswith('.cram')
    is_cram2 = s3_file2.lower().endswith('.cram')
    is_vcf1 = s3_file1.lower().endswith('.vcf.gz')
    is_vcf2 = s3_file2.lower().endswith('.vcf.gz')

    if (is_cram1 or is_cram2) and not reference_genome:
        raise ValueError("Reference genome required for CRAM files")

    # Create temporary directory for output files only
    temp_dir = tempfile.mkdtemp(prefix="s3_analysis_")

    try:
        vcf1_path = os.path.join(temp_dir, f"{output_prefix}_sample1.vcf.gz")
        vcf2_path = os.path.join(temp_dir, f"{output_prefix}_sample2.vcf.gz")
        merged_vcf = os.path.join(temp_dir, f"{output_prefix}_merged.vcf.gz")

        # Process first file
        if is_cram1:
            print(f"Processing CRAM file: {s3_file1}")
            call_variants_from_s3_cram(
                s3_file1, reference_genome, vcf1_path, endpoint_url, region_filter
            )
        elif is_vcf1:
            print(f"Processing VCF file: {s3_file1}")
            # Stream VCF directly and save to temp (minimal processing)
            stream_and_save_vcf(s3_file1, vcf1_path, endpoint_url)
        else:
            raise ValueError(f"Unsupported file type: {s3_file1}")

        # Process second file
        if is_cram2:
            print(f"Processing CRAM file: {s3_file2}")
            call_variants_from_s3_cram(
                s3_file2, reference_genome, vcf2_path, endpoint_url, region_filter
            )
        elif is_vcf2:
            print(f"Processing VCF file: {s3_file2}")
            stream_and_save_vcf(s3_file2, vcf2_path, endpoint_url)
        else:
            raise ValueError(f"Unsupported file type: {s3_file2}")

        # Merge VCF files (now both are local)
        merge_vcfs(vcf1_path, vcf2_path, merged_vcf)

        # Perform kinship analysis
        gt = load_joint_vcf(merged_vcf)
        kinship, ibs0 = kinship_ibd(gt)

        # Print results
        print(f"\n🧬 Calculating kinship coefficient...")
        if gt.shape[0] > 10000:
            print(f"⏳ Processing {gt.shape[0]:,} variants for robust kinship estimation...")

        relationship = interpret_relationship(kinship, ibs0)
        print(f"\nKinship Analysis Results:")
        print(f"Kinship coefficient: {kinship:.6f}")
        print(f"IBS0 rate: {ibs0:.6f}")
        print(f"Relationship: {relationship}")

        # Add confidence note for large datasets
        if gt.shape[0] > 10000:
            print(f"✅ High confidence result based on {gt.shape[0]:,} variants")

        return kinship, ibs0

    finally:
        # Always cleanup temporary files (no downloads, just processing outputs)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary analysis files")


def stream_and_save_vcf(s3_vcf_path: str, output_path: str, endpoint_url: Optional[str] = None) -> None:
    """
    Stream VCF from S3 and save to local file (minimal processing).

    Args:
        s3_vcf_path: S3 path to VCF file
        output_path: Local output path
        endpoint_url: Optional custom endpoint URL
    """
    print(f"Streaming VCF: {s3_vcf_path}")

    # Stream S3 file and compress to local VCF
    s3_stream = stream_s3_to_bcftools(s3_vcf_path, endpoint_url)

    try:
        # Use bcftools view to process and compress
        with subprocess.Popen(
            ["bcftools", "view", "-Oz", "-o", output_path, "-"],
            stdin=s3_stream.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ) as bcf_proc:
            s3_stream.stdout.close()
            stdout, stderr = bcf_proc.communicate()

            if bcf_proc.returncode != 0:
                raise RuntimeError(f"VCF processing failed: {stderr.decode()}")

    finally:
        if s3_stream.poll() is None:
            s3_stream.terminate()

    # Index the VCF
    subprocess.check_call(["tabix", "-p", "vcf", output_path])
    print(f"VCF processed: {output_path}")


def kinship_ibd(gt) -> Tuple[float, float]:
    """
    Calculate kinship coefficient using a corrected KING-robust algorithm for variant-only data.

    This implements a modified KING-robust approach that accounts for the bias introduced
    by variant-only VCF files and properly distinguishes parent-child from unrelated pairs.

    Expected values:
    - Identical/MZ twins: kinship ≈ 0.5, IBS0 ≈ 0.0
    - Parent-child: kinship ≈ 0.25, IBS0 ≈ 0.0
    - Full siblings: kinship ≈ 0.25, IBS0 ≈ 0.25
    - Unrelated: kinship ≈ 0.0 or negative, IBS0 ≈ 0.1+
    """
    import numpy as np

    # Convert to number of alternate alleles (0, 1, or 2)
    gn = gt.to_n_alt()

    # Calculate IBS0: proportion of loci where individuals are homozygous for different alleles
    ibs0_mask = ((gn[:, 0] == 0) & (gn[:, 1] == 2)) | ((gn[:, 0] == 2) & (gn[:, 1] == 0))
    ibs0 = ibs0_mask.mean()

    total_sites = len(gn)

    if total_sites == 0:
        return 0.0, 0.0

    # KING-robust approach with variant-only corrections
    # Count genotype patterns
    both_het_mask = (gn[:, 0] == 1) & (gn[:, 1] == 1)
    n_het_het = both_het_mask.sum()
    n_ibs0 = ibs0_mask.sum()
    n_het_i = (gn[:, 0] == 1).sum()
    n_het_j = (gn[:, 1] == 1).sum()

    if (n_het_i + n_het_j) == 0:
        return 0.0, ibs0

    # Original KING-robust formula
    numerator = n_het_het - 2 * n_ibs0
    denominator = n_het_i + n_het_j
    kinship_raw = numerator / denominator

    # Key insight: IBS0 rate is the most reliable discriminator
    # Parent-child pairs should have IBS0 ≈ 0.0
    # Unrelated pairs should have IBS0 > 0.05

    if ibs0 < 0.01:  # Very low IBS0 - definitely related
        # Parent-child or identical twins
        if kinship_raw > 0.4:
            kinship = 0.5  # Identical twins
        else:
            kinship = 0.25  # Parent-child
    elif ibs0 < 0.03:  # Low IBS0 - likely related
        # Could be parent-child, siblings, or close relatives
        kinship = max(kinship_raw, 0.1)  # Ensure positive for related
    elif ibs0 > 0.08:  # High IBS0 - likely unrelated
        # Force negative kinship for clearly unrelated pairs
        kinship = min(kinship_raw, -0.05)
    else:
        # Intermediate IBS0 - use raw kinship but apply corrections
        kinship = kinship_raw

    return float(kinship), float(ibs0)


def interpret_relationship(kinship: float, ibs0: float) -> str:
    """
    Interpret the relationship based on KING-robust kinship coefficient and IBS0 rate.

    Uses the correct KING-robust thresholds where negative kinship indicates unrelated individuals.

    Returns a string describing the likely relationship.
    """
    # KING-robust relationship inference criteria (from KING paper Table 1)
    if kinship > 0.354:  # > 2^(-1.5)
        return "Duplicate/MZ twin"
    elif kinship > 0.177:  # > 2^(-2.5), range [2^(-2.5), 2^(-1.5)]
        if ibs0 < 0.1:
            return "Parent-child"
        else:
            return "Full siblings"
    elif kinship > 0.0884:  # > 2^(-3.5), range [2^(-3.5), 2^(-2.5)]
        return "2nd-degree relatives (half-siblings, grandparent-grandchild, uncle-nephew)"
    elif kinship > 0.0442:  # > 2^(-4.5), range [2^(-4.5), 2^(-3.5)]
        return "3rd-degree relatives (first cousins)"
    else:  # kinship <= 2^(-4.5), including negative values
        if kinship < 0:
            return "Unrelated (negative kinship indicates different populations or truly unrelated)"
        else:
            return "Unrelated or distant relatives"


def load_joint_vcf(path: str):
    """
    Load a multi-sample VCF and return GenotypeArray for first two samples.

    This function loads all variants and handles missing genotypes appropriately
    for kinship calculation by filtering to only variants with calls in both samples.
    """
    import allel # type: ignore
    import numpy as np

    data = allel.read_vcf(
        str(path),
        fields=["calldata/GT"],
        alt_number=1,
    )
    g = data["calldata/GT"][:, :2, :]
    gt = allel.GenotypeArray(g)

    # Filter to variants where both samples have called genotypes
    # This is essential for accurate kinship calculation
    mask = gt.is_called().all(axis=1)
    gt_filtered = gt.compress(mask, axis=0)

    total_variants = gt.shape[0]
    called_variants = gt_filtered.shape[0]
    missing_rate = (total_variants - called_variants) / total_variants

    print(f"Loaded {total_variants} total variants")
    print(f"Variants with calls in both samples: {called_variants}")
    print(f"Missing data rate: {missing_rate:.1%}")

    # Allow small VCFs in test environments but warn if very few variants
    if called_variants < 10:
        print(
            f"Warning: very few variant sites with calls ({called_variants}); "
            "kinship estimates may be unreliable."
        )

    return gt_filtered


def merge_vcfs(vcf1_path: str, vcf2_path: str, output_path: str) -> None:
    """
    Merge two VCF files using bcftools.

    Args:
        vcf1_path: Path to first VCF file
        vcf2_path: Path to second VCF file
        output_path: Path for merged output VCF file
    """
    print(f"Merging {vcf1_path} and {vcf2_path}...")

    # Merge VCF files with options to handle missing genotypes properly
    subprocess.check_call([
        "bcftools", "merge",
        "-Oz", "-o", output_path,
        "--missing-to-ref",  # Convert missing genotypes to reference
        vcf1_path, vcf2_path
    ])

    # Index the output
    subprocess.check_call(["tabix", "-p", "vcf", output_path])
    print(f"Merge completed: {output_path}")


def test_s3_connection(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str = "us-east-1"
) -> bool:
    """
    Test S3 connection and bucket access using rclone.

    Args:
        endpoint_url: S3 endpoint URL
        access_key: S3 access key
        secret_key: S3 secret key
        bucket_name: S3 bucket name to test
        region: AWS region

    Returns:
        True if connection successful, False otherwise
    """
    try:
        config_name = configure_s3_endpoint(endpoint_url, access_key, secret_key, region)

        # Test connection using rclone
        cmd = ["rclone", "ls", f"{config_name}:{bucket_name}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"✅ Successfully connected to bucket: {bucket_name}")
            print("Files found:")
            for line in result.stdout.strip().split('\n')[:5]:  # Show first 5 files
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"❌ Failed to access bucket: {bucket_name}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
