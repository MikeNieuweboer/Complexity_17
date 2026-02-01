# Gen AI usage
This file will outline the use of GenAI in this project. For each relevant file an entry will be made that specifies where GenAI is used in the file, including the prompt used. Additionally some remarks can be given at each entry on experiences while using GenAI in this way.

## Template Entry
- File: <rel_path_to_file>
    - Use of GenAI
        - Details
        - Prompt
    - Remarks

## Entries

- nn.py <src/nn.py>
    - Use of GenAI
        - Gemini 3.0 was used to generate docstrings for the functionality in this file. This was done for each function individually using the inline chat. 
        - __Prompt:__ Analyze the specific function and create a consise docstring
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - None

- plotting.py <src/plotting.py>
    - Use of GenAI
        - Gemini 3.0 was used to generate docstrings for the functionality in this file. This was done for each function individually using the inline chat. 
        - __Prompt:__ Analyze the specific function and create a consise docstring
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - None

- train.py <src/train.py>
    - Use of GenAI
        - Gemini 3.0 was used to generate docstrings for the functionality in this file. This was done for each function individually using the inline chat. 
        - __Prompt:__ Analyze the specific function and create a consise docstring
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - The importance of manually reading the generated docstrings was stressed, because the generated docstring for the train() function contained some examples that were not applicable to our use case. One example was that in the description of the n_channels argument it gave as an example (RGBA + hidden), whereas for our project we only have (alpha + hidden).
