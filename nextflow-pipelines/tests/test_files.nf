#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Exercise 2: Working with files
process countLines {
    tag "$file"
    
    input:
    path file

    output:
    tuple path(file), stdout

    script:
    """
    wc -l $file | awk '{print \$1}'
    """
}

workflow {
    // Create channel with sample data
    Channel.of(
        ['sample1', 'Test line 1'],
        ['sample2', 'Test line 2\nAnother line'],
        ['sample3', 'Single line'],
        ['sample4', 'Line 1\nLine 2\nLine 3']
    )
    .map { name, content ->
        def file = file("${workDir}/test_${name}.txt")
        file.text = content
        return file
    }
    .set { test_files }
    
    // Count lines in each file
    countLines(test_files)
        | map { file, count -> "${file.name}: ${count.trim()} lines" }
        | view
}
