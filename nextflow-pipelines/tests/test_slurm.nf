#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Exercise 4: SLURM submission with resource labels
process lightTask {
    label 'process_low'
    tag "Light-$x"
    
    input:
    val x

    output:
    stdout

    script:
    """
    echo "Light task $x running with \${SLURM_CPUS_PER_TASK:-local} CPUs"
    sleep 5
    echo "Task $x completed"
    """
}

process heavyTask {
    label 'process_high'
    tag "Heavy-$x"
    
    input:
    val x

    output:
    stdout

    script:
    """
    echo "Heavy task $x running with \${SLURM_CPUS_PER_TASK:-local} CPUs"
    sleep 10
    echo "Heavy task $x completed"
    """
}

workflow {
    // Run light tasks
    Channel.of(1, 2, 3) | lightTask | view { "Light: $it" }
    
    // Run heavy tasks
    Channel.of('A', 'B') | heavyTask | view { "Heavy: $it" }
}

// Note: Resource labels are defined in config/base.config
// process_low: 2 CPUs, 4 GB
// process_high: 8 CPUs, 16 GB
