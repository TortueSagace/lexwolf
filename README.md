# Microprocessors

## Example of Library Use

`````python
from microprocessors.architechture import RiSC16

risc = RiSC16()

# Properly queue instructions
risc.encode_instruction('ADDI', 1, 0, 5)  # R1 = 5
risc.encode_instruction('LUI', 2, None, 1)  # R2 = 1 << (16 - immediate_bits)
risc.encode_instruction('ADD', 3, 1, 2)  # R3 = R1 + R2
risc.encode_instruction('NAND', 4, 1, 2)  # R4 = ~(R1 & R2)
risc.encode_instruction('SW', 3, 0, 10)  # Memory[10] = R3
risc.encode_instruction('LW', 5, 0, 10)  # R5 = Memory[10]
risc.encode_instruction('BEQ', 1, 2, 2)  # if R1 == R2, skip next 2 instructions
risc.encode_instruction('ADDI', 6, 0, 1)  # R6 = 1 (skipped if R1 == R2)
risc.encode_instruction('JALR', 7, 0)  # R7 = PC + 1; PC = R0

# Run all and visalize register at final state
risc.run_program()
risc.visualize_state()
`````

## Documentation
Refer to the in-line comments and method docstrings for detailed usage of each feature.

## Contribution
Contributions are welcome! Feel free to submit pull requests, suggest features, or report bugs.

## License
This library is distributed under the MIT license. See `LICENSE` for more information.

## Contact
- **Author**: Alexandre Le Mercier
- **Date**: February 20, 2024
- **Email**: [alexandre.le.mercier@ulb.be](mailto:alexandre.le.mercier@ulb.be)
- **LinkedIn**: [Alexandre Le Mercier](https://www.linkedin.com/in/alexandre-le-mercier-7b5594283/details/experience/)

Happy coding!