# QUDsim: Quantifying Discourse Similarities in LLM-Generated Text
Authors: Ramya Namuduri, Yating Wu, Asher Zheng, Manya Wadhwa, Jessy Li, Greg Durrett

Compare documents (stories, articles, blog posts, or even obituaries) and see how similar their discourse progressions are. Discourse progression refers to the backbone structure of a text (for example, tropes in stories). 

This repository contains details on the entire process of generating and using QUDs (Questions Under Discussion) to find discourse similarities.

Check out our paper on QUDsim for more!

## Installation/Setup

1. Create a virtual environment (for ex:)
   ~~~
   python -m venv [name_of_env]
   ~~~
3. Set the OpenAI key as an environment variable (OPENAI_API_KEY)
4. Install dependencies
   ~~~
   pip install -r requirements.txt
   ~~~

## How to...
To use QUDsim to align documents and find discourse similarities, run:
~~~
python qudsim.py --query_file [name_of_query_doc.json]
~~~
Note: if no file is specified, the example will query file (```example_query.json```) will automatically be used.

The input file passed in must either be a ```.json``` or ```.csv```. The dataframe needs to contain two columns for each document being compared, and two columns for the model that generated the document. Refer to ```PairColumns``` in ```schema.py``` for naming the columns.

### Additional Configuration
Additional flags can either be passed as arguments or through ```config.py```. You can alter: 
1. ```--level```: The level of abstraction of the QUDs with 0 being content-specific and 1 being generic.
2. ```qg_gpt_model```: The model used for the QUD generation process
3. ```qa_gpt_model```: The model used to align documents through answerability
4. ```with_replacement```: Setting this to ```False``` will treat each pair of documents independently even if the same document appears in other queries (i.e. a new set of QUDs is generated for every instance of the document)

## Dataset
1. ```dataset/qud_data.json```: The dataset we built and ran experiments on.
2. ```dataset/annotations.json```: Annotations we collected during intrinsic evaluation.
3. ```dataset/similarity.json```: QUDsim scores and document alignments for our dataset.

### Format
```DatasetColumns``` and ```SimilarityColumns``` in ```schema.py``` contains information on how the datasets are structured and how they can be accessed. 
