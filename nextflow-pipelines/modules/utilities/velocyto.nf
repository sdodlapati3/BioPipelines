module velocyto {

    /*
     * Velocyto module for processing RNA velocity analysis
     * using Nextflow DSL2 best practices.
     */

    version '1.0'

    process runVelocyto {

        tag "${sample_id}"

        // Define the container to use for this process
        container "${params.containers.scrna-seq}"

        // Define the input parameters
        input:
            tuple val(sample_id), path(bam_file), path(gtf_file)
            path whitelist_file
            path index_file
            path genome_file

        // Define the output files
        output:
            path "${sample_id}.loom" into loom_files

        // Define the script to execute
        script:
        """
        velocyto run \
            -b ${whitelist_file} \
            -o . \
            -m ${index_file} \
            ${bam_file} \
            ${gtf_file} \
            ${genome_file}
        """
    }
}