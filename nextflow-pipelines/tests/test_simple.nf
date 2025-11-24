#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Exercise 1: Simple process with value input
process sayHello {
    input:
    val name

    output:
    stdout

    script:
    """
    echo "Hello, $name!"
    """
}

workflow {
    // Create a channel with names
    names = Channel.of('Alice', 'Bob', 'Charlie')
    
    // Run the process
    sayHello(names) | view
}
