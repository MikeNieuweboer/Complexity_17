# Gen AI usage
This file will outline the use of GenAI in this project. For each relevant file an entry will be made that specifies where GenAI is used in the file, including the prompt used. Additionally some remarks can be given at each entry on experiences while using GenAI in this way.

## Template Entry
- File: <rel_path_to_file>
    - Use of GenAI
        - Details
        - Prompt
    - Remarks

## Entries
- ea.py <src/ea.py>
    - Use of GenAI
        - Haiku 4.5 was used to generate docstrings for the functionality in this file
        - __Prompt:__ Analyse the functionality of @src/ea.py, assuming that the rest of the codebase is implemented and write the required docstrings for the classes and functions within.
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - Read everything to be sure the correct comments were generated, but it looked good to me.

- mcaf.py <src/mcaf.py>
    - Use of GenAI
        - Haiku 4.5 was used to generate docstrings for the functionality in this file
        - __Prompt:__
            - Analyse the @src/mcaf.py file to allow for future doc comment generation.
            - Generate the PEP styled doc comments.
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - Read everything to be sure the correct comments were generated, but it looked good to me.

- plot_mcaf.py <src/plot_mcaf.py>
    - Use of GenAI
        - Haiku 4.5 was used to generate docstrings for the functionality in this file
        - __Prompt:__
            - Analyse the @src/plot_mcaf.py file and try to understand its structure for future doc comment generation.
            - Generate the PEP style comment.
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - Read everything to be sure the correct comments were generated, but it looked good to me.

- nn.py <src/nn.py>
    - Use of GenAI
        - Gemini 3.0 was used to generate docstrings for the functionality in this file. This was done for each function individually using the inline chat. 
        - __Prompt:__ Analyze the specific function and create a concise docstring
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - None

- plotting.py <src/plotting.py>
    - Use of GenAI
        - Gemini 3.0 was used to generate docstrings for the functionality in this file. This was done for each function individually using the inline chat. 
        - __Prompt:__ Analyze the specific function and create a concise docstring
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - None

- train.py <src/train.py>
    - Use of GenAI
        - Gemini 3.0 was used to generate docstrings for the functionality in this file. This was done for each function individually using the inline chat. 
        - __Prompt:__ Analyze the specific function and create a concise docstring
        - Afterwards manual changes were made to the liking of the coder.
    - Remarks
        - The importance of manually reading the generated docstrings was stressed, because the generated docstring for the train() function contained some examples that were not applicable to our use case. One example was that in the description of the n_channels argument it gave as an example (RGBA + hidden), whereas for our project we only have (alpha + hidden).

- grid.py <src/train.py>
    - Use of GenAI
        - ChatGPT 5.2 was used to generate docstrings for the functionality in this file. This was done for the whole file in one prompt.
        - __Prompt:__ Add a docstring for each function.
        - Afterwards manual changes were made to the liking of the coder.

        - ChatGPT 5.2 was used to generate sanity checks for functions where its functionality depends on previous function calls.
        - __Prompt:__ Add a sanity check to this function that checks [x].
    - Remarks
        - None