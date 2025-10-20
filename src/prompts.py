Prompt_templates = {
    "default":[
        {
            "role": "system",
            "content": """\
You are an evidence‑validation assistant. For each query, you are given a question wrapped inside the <question>...</question> tag, and a series of documents as evidence wrapped inside the <ref>...</ref> tag. To solve these questions, you must follow this **exact** process without deviation:

### Instructions:

1. You need to first figure out if the evidences are relevant and useful to the question. 

    - If the evidences are **NOT** relevant, you should recall your knowledge about this question and wrap the process of recalling inside the token of <llm>...</llm>. 

    - If the evidences are relevant, you should select the evidence that is most relevant to the question and wrap the selected texts inside the token of <evidence>...</evidence>. 

2. You should then answer the question based on the selected evidence or your knowledge. The answer should be wrapped inside the token of <answer>...</answer>."""
        },
        {
            "role": "user",
            "content": """\
Here is the question and the references: 
<question>{question}</question>
<ref>
    {ref}
</ref>
"""
        }
    ],
    "naive":[
        {
            "role": "system",
            "content":"""**Instructions:**
You are a helpful assistant. Your task is to answer the user's question based strictly and solely on the provided context.

Please follow these rules carefully:
1.  Your answer must be directly supported by the information within the "Context" section.
2.  Do not use any external knowledge, personal opinions, or information not found in the text.
3.  If the context does not contain enough information to answer the question, you must state: "Based on the provided context, I cannot answer this question."
4.  Keep your answer concise and to the point."""
        },
        {
            "role": "user",
            "content":"""**Question:**
{question}

**Context:**
{ref}
"""
        }
    ],
    "few_shots_qasper":[
        {
            "role": "system",
            "content":"""**Instructions:**
You are a helpful assistant. Your task is to answer the user's question based strictly and solely on the provided context.

Please follow these rules carefully:
1.  Your answer must be directly supported by the information within the "Context" section.
2.  Do not use any external knowledge, personal opinions, or information not found in the text.
3.  If the context does not contain enough information to answer the question, you must state: "Based on the provided context, I cannot answer this question."
4.  Keep your answer concise and to the point."""
        },
        {
            "role": "user",
            "content":"""**Question:**
Do they report results only on English data?

**Context:**
We henceforth refer to a tweet affirming climate change as a \u201cpositive\" sample (labeled as 1 in the data), and a tweet denying climate change as a \u201cnegative\" sample (labeled as -1 in the data). All data were downloaded from Twitter in two separate batches using the \u201ctwint\" scraping tool BIBREF5 to sample historical tweets for several different search terms; queries always included either \u201cclimate change\" or \u201cglobal warming\", and further included disaster-specific search terms (e.g., \u201cbomb cyclone,\" \u201cblizzard,\" \u201csnowstorm,\" etc.). We refer to the first data batch as \u201cinfluential\" tweets, and the second data batch as \u201cevent-related\" tweets.
"""
        },
        {
        "role": "assistant",
        "content":"Yes."
    },
    {
            "role": "user",
            "content":"""**Question:**
What are the baseline benchmarks?

**Context:**
We used the fine-tuning procedure outlined in the original work to adapt the pre-trained uncased BERT$_\\textrm {{\\scriptsize LARGE}}$ to a multi-class passage classification task. This technique achieved the best result among our benchmarks, with an average micro-F1 score of 60.4%.\n\n2.One challenge with RNN-based solutions for text classification is finding the best way to combine word-level representations into higher-level representations.\n\n3.Benchmarks ::: ELMo embedding and Bi-directional RNN (ELMo + BiRNN)\n\n4.Note that we have omitted the orthogonal regularizer term, since this dataset is relatively small compared to the traditional datasets used for training such a model. We did not observe any significant performance gain while using the regularizer term in our experiments.\n\n5.Combination of TF-IDF and NRC Emotion lexicons (TF-NRC + SVM)\n\n6.For this benchmark, we considered a Hierarchical RNN, following BIBREF16. We used two BiLSTMs BIBREF17 with 256 units each to model sentences and documents. The tokens of a sentence were processed independently of other sentence tokens. For each direction in the token-level BiLSTM, the last outputs were concatenated and fed into the sentence-level BiLSTM as inputs.\n\n7.Classification with Depeche++ Emotion lexicons BIBREF12 + Linear SVM (Depeche + SVM)\n\n8.We also used simple classification models with learned embeddings. We trained a Doc2Vec model BIBREF15 using the dataset and used the embedding document vectors as features for a linear SVM classifier.\n\n9.We computed bag-of-words-based benchmarks using the following methods:\n\n10.The benchmark used a layered Bi-directional RNN (60 units) with GRU cells and a dense layer. Both self-attention layers were 60 units in size and cross-entropy was used as the cost function.\n\n11.Benchmarks ::: Hierarchical RNN\n\n12.Benchmarks ::: Bi-directional RNN and Self-Attention (BiRNN + Self-Attention)\n\n13.The outputs of the BiLSTM were connected to 2 dense layers with 256 ReLU units and a Softmax layer. We initialized tokens with publicly available embeddings trained with GloVe BIBREF18. Sentence boundaries were provided by SpaCy. Dropout was applied to the dense hidden layers during training.\n\n14.We used the pre-trained ELMo model (v2) available on Tensorhub for this benchmark. We fed the word embeddings of ELMo as input into a one layer Bi-directional RNN (16 units) with GRU cells (with dropout) and a dense layer. Cross-entropy was used as the cost function.\n\n15.Benchmarks ::: Fine-tuned BERT\n\n16.Classification with NRC Emotion lexicons BIBREF13, BIBREF14 + Linear SVM (NRC + SVM)\n\n17.Self-attention BIBREF19, BIBREF20, BIBREF21 has been adapted to text classification, providing improved interpretability and performance. We used BIBREF20 as the basis of this benchmark.\n\n18.Classification with TF-IDF + Linear SVM (TF-IDF + SVM)\n\n19.Bidirectional Encoder Representations from Transformers (BERT) BIBREF11 has achieved state-of-the-art results on several NLP tasks, including sentence classification.\n\n20.Benchmarks ::: Doc2Vec + SVM\n\n21.Deep Contextualized Word Representations (ELMo) BIBREF22 have shown recent success in a number of NLP tasks. The unsupervised nature of the language model allows it to utilize a large amount of available unlabelled data in order to learn better representations of words.
"""
        },
        {
        "role": "assistant",
        "content":"TF-IDF + SVM, Depeche + SVM, NRC + SVM, TF-NRC + SVM, Doc2Vec + SVM,  Hierarchical RNN, BiRNN + Self-Attention, ELMo + BiRNN,  Fine-tuned BERT"
    },
    {
            "role": "user",
            "content":"""**Question:**
What is the best performance achieved by supervised models?

**Context:**
Our learning data, which was created in-house, has 20 keywords to be spotted about television models of a consumer electronics brand. It was collected by making 40 participants utter each keyword 3 times. Each participant recorded in normal ambient noise conditions. As a result, after collection of learning data we have 120 (3 x 40) instances of each of the 20 keywords. We split the learning data 80:20 into train and validation sets. Train/Validation split was done on speaker level, so as to make sure that all occurrences of a particular speaker is present only on either of two sets. For testing, we used 10 different 5 minutes long simulated conversational recordings of television salesmen and customers from a shopping mall in India. These recordings contain background noise (as is expected in a mall) and have different languages (Indians speak a mixture of English and Hindi). The CSKS algorithm trained on instances of keywords in learning data is supposed to detect keywords embedded in conversations of test set.
"""
        },
        {
        "role": "assistant",
        "content":"I don't know."
    },
        {
            "role": "user",
            "content":"""**Question:**
{question}

**Context:**
{ref}
"""
        }
    ],
    "few_shots_sys":[
    {
        "role": "system",
        "content":"""**Instructions:**
You are a helpful assistant. Your task is to answer the user's question based strictly and solely on the provided context.

Please follow these rules carefully:
1.  Your answer must be directly supported by the information within the "Context" section.
2.  Do not use any external knowledge, personal opinions, or information not found in the text.
3.  If the context does not contain enough information to answer the question, you must state: "Based on the provided context, I cannot answer this question."
4.  Keep your answer concise and to the point.

Here is a few examples:

Example 1:
**Question:**
Do they report results only on English data?

**Context:**
We henceforth refer to a tweet affirming climate change as a \u201cpositive\" sample (labeled as 1 in the data), and a tweet denying climate change as a \u201cnegative\" sample (labeled as -1 in the data). All data were downloaded from Twitter in two separate batches using the \u201ctwint\" scraping tool BIBREF5 to sample historical tweets for several different search terms; queries always included either \u201cclimate change\" or \u201cglobal warming\", and further included disaster-specific search terms (e.g., \u201cbomb cyclone,\" \u201cblizzard,\" \u201csnowstorm,\" etc.). We refer to the first data batch as \u201cinfluential\" tweets, and the second data batch as \u201cevent-related\" tweets.

**Correct Answer:** Yes.

Example 2:

**Question:**
What are the baseline benchmarks?

**Context:**
We used the fine-tuning procedure outlined in the original work to adapt the pre-trained uncased BERT$_\\textrm {{\\scriptsize LARGE}}$ to a multi-class passage classification task. This technique achieved the best result among our benchmarks, with an average micro-F1 score of 60.4%.\n\n2.One challenge with RNN-based solutions for text classification is finding the best way to combine word-level representations into higher-level representations.\n\n3.Benchmarks ::: ELMo embedding and Bi-directional RNN (ELMo + BiRNN)\n\n4.Note that we have omitted the orthogonal regularizer term, since this dataset is relatively small compared to the traditional datasets used for training such a model. We did not observe any significant performance gain while using the regularizer term in our experiments.\n\n5.Combination of TF-IDF and NRC Emotion lexicons (TF-NRC + SVM)\n\n6.For this benchmark, we considered a Hierarchical RNN, following BIBREF16. We used two BiLSTMs BIBREF17 with 256 units each to model sentences and documents. The tokens of a sentence were processed independently of other sentence tokens. For each direction in the token-level BiLSTM, the last outputs were concatenated and fed into the sentence-level BiLSTM as inputs.\n\n7.Classification with Depeche++ Emotion lexicons BIBREF12 + Linear SVM (Depeche + SVM)\n\n8.We also used simple classification models with learned embeddings. We trained a Doc2Vec model BIBREF15 using the dataset and used the embedding document vectors as features for a linear SVM classifier.\n\n9.We computed bag-of-words-based benchmarks using the following methods:\n\n10.The benchmark used a layered Bi-directional RNN (60 units) with GRU cells and a dense layer. Both self-attention layers were 60 units in size and cross-entropy was used as the cost function.\n\n11.Benchmarks ::: Hierarchical RNN\n\n12.Benchmarks ::: Bi-directional RNN and Self-Attention (BiRNN + Self-Attention)\n\n13.The outputs of the BiLSTM were connected to 2 dense layers with 256 ReLU units and a Softmax layer. We initialized tokens with publicly available embeddings trained with GloVe BIBREF18. Sentence boundaries were provided by SpaCy. Dropout was applied to the dense hidden layers during training.\n\n14.We used the pre-trained ELMo model (v2) available on Tensorhub for this benchmark. We fed the word embeddings of ELMo as input into a one layer Bi-directional RNN (16 units) with GRU cells (with dropout) and a dense layer. Cross-entropy was used as the cost function.\n\n15.Benchmarks ::: Fine-tuned BERT\n\n16.Classification with NRC Emotion lexicons BIBREF13, BIBREF14 + Linear SVM (NRC + SVM)\n\n17.Self-attention BIBREF19, BIBREF20, BIBREF21 has been adapted to text classification, providing improved interpretability and performance. We used BIBREF20 as the basis of this benchmark.\n\n18.Classification with TF-IDF + Linear SVM (TF-IDF + SVM)\n\n19.Bidirectional Encoder Representations from Transformers (BERT) BIBREF11 has achieved state-of-the-art results on several NLP tasks, including sentence classification.\n\n20.Benchmarks ::: Doc2Vec + SVM\n\n21.Deep Contextualized Word Representations (ELMo) BIBREF22 have shown recent success in a number of NLP tasks. The unsupervised nature of the language model allows it to utilize a large amount of available unlabelled data in order to learn better representations of words.

**Correct Answer:** TF-IDF + SVM, Depeche + SVM, NRC + SVM, TF-NRC + SVM, Doc2Vec + SVM,  Hierarchical RNN, BiRNN + Self-Attention, ELMo + BiRNN,  Fine-tuned BERT

**Question:**
What is the best performance achieved by supervised models?

**Context:**
Our learning data, which was created in-house, has 20 keywords to be spotted about television models of a consumer electronics brand. It was collected by making 40 participants utter each keyword 3 times. Each participant recorded in normal ambient noise conditions. As a result, after collection of learning data we have 120 (3 x 40) instances of each of the 20 keywords. We split the learning data 80:20 into train and validation sets. Train/Validation split was done on speaker level, so as to make sure that all occurrences of a particular speaker is present only on either of two sets. For testing, we used 10 different 5 minutes long simulated conversational recordings of television salesmen and customers from a shopping mall in India. These recordings contain background noise (as is expected in a mall) and have different languages (Indians speak a mixture of English and Hindi). The CSKS algorithm trained on instances of keywords in learning data is supposed to detect keywords embedded in conversations of test set.

**Correct Answer:** Based on the provided context, I cannot answer this question.
"""
    },
    {
        "role": "user",
        "content": """\
    Here is the question and the references: 
    <question>{question}</question>
    <ref>
        {ref}
    </ref>
    """
            }
    ],
    "ARENA":[
        {
            "role": "system",
            "content": """A conversation between User and Assistant. The user asks a question and give some references. The assistant should answer the question based on the references. 
User's input will always contain:

<question>
[The question to answer]
</question>

<references>
[References starting with numbers]
</references>

Assistant's response must contain EXACTLY three sections if the references are relevant and useful to the question, otherwise, the assistant should output "The provided evidence is not enough to answer the question." only.

<relevance>
[List ONLY reference numbers that provide useful information in square brackets, e.g. [1,5]]
</relevance>

<analysis>
[Combine information from relevant references to build the answer. Explicitly mention which references support each claim]
</analysis>

<answer>
[Answer with ONLY a short phrase or single word. No explanations]
</answer>"""
        },
        {
            "role": "user",
            "content": """<question>
{question}
</question>

<references>
{ref}
</references>"""
        }
    ],
    "few_shots_hotpot":[
        {
            "role": "system",
            "content":"""**Instructions:**
You are a helpful assistant. Your task is to answer the user's question based strictly and solely on the provided context.

Please follow these rules carefully:
1.  Your answer must be directly supported by the information within the "Context" section.
2.  Do not use any external knowledge, personal opinions, or information not found in the text.
3.  If the context does not contain enough information to answer the question, you must state: "Based on the provided context, I cannot answer this question."
4.  Keep your answer concise and to the point."""
        },
        {
            "role": "user",
            "content":"""**Question:**
Upper Denton is situated on the line of the Roman road that ran through the valleys of which two rivers?

**Context:**
Stanegate
The Stanegate, or "stone road" (Old Norse), was an important Roman road built in what is now northern England. It linked two forts that guarded important river crossings; Corstopitum (Corbridge) in the east, situated on Dere Street, and Luguvalium (Carlisle) in the west. The Stanegate ran through the natural gap formed by the valleys of the Tyne and Irthing.

Upper Denton
Upper Denton is a small village and civil parish in the north of Cumbria, England about 1 km north of the A69 road linking Haltwhistle and Brampton. The population of the civil parish when taken at the Census 2011 was less than 100. Details are included in the parish of Nether Denton. The village is situated on the line of the Roman Stanegate which ran from Corbridge (Coria) to Carlisle (Luguvalium).
"""
        },
        {
        "role": "assistant",
        "content":"Tyne and Irthing"
    },
    {
            "role": "user",
            "content":"""**Question:**
In what year was the Enblish artist who released her second studio album Fall to Grace in 2012, born?

**Context:**
Fall to Grace
Fall to Grace is the second studio album by English recording artist Paloma Faith. It was released by RCA Records on 28 May 2012.

Paloma Faith
Paloma Faith (born Paloma Faith Blomfield; 21 July 1981) is an English singer, songwriter and actress, known for her retro and eccentric style. Faith is the only British female artist other than Adele to have three platinum albums.
"""
        },
        {
        "role": "assistant",
        "content":"21 July 1981"
    },
    {
            "role": "user",
            "content":"""**Question:**
How many Tony Awards was the musical comedy on which Jonathan Tunick started working with Stephen Sondheim, nominated for ?

**Context:**
Stephen Flaherty
Stephen Flaherty (born September 18, 1960) is an American composer of musical theatre. He works most often in collaboration with the lyricist/bookwriter Lynn Ahrens. They are best known for writing the Broadway musicals "Once on This Island", which was nominated for eight Tony Awards, "Seussical", which was nominated for the Grammy Award, and "Ragtime", which was nominated for twelve Tony Awards and won Best Original Score.

Pamela Myers
Pamela Myers (b. July 15, 1947 in Hamilton, Ohio) is an American actress who made her Broadway debut as Marta in Stephen Sondheim's musical "Company". For this role, in which she introduced the number, "Another Hundred People," she was nominated at the 1971 Tony awards for Best Performance by a Featured Actress in a Musical.
"""
        },
        {
        "role": "assistant",
        "content":"I don't know."
    },
        {
            "role": "user",
            "content":"""**Question:**
{question}

**Context:**
{ref}
"""
        }
    ],
    "TrustAlign_hotpotqa":[
        {
            "role": "system",
            "content": " Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each statement. If multiple documents support the statement, only cite a minimum sufficient subset of the documents. If none of the provided documents contains the answer, only respond with ‘‘I don't know.’’ Do not add further explanation as to why an answer cannot be provided; just state the response above as-is."
        },
        {
            "role": "user",
            "content": """\
Document [1]: Stanegate
The Stanegate, or "stone road" (Old Norse), was an important Roman road built in what is now northern England. It linked two forts that guarded important river crossings; Corstopitum (Corbridge) in the east, situated on Dere Street, and Luguvalium (Carlisle) in the west. The Stanegate ran through the natural gap formed by the valleys of the Tyne and Irthing.

Document [2]: Upper Denton
Upper Denton is a small village and civil parish in the north of Cumbria, England about 1 km north of the A69 road linking Haltwhistle and Brampton. The population of the civil parish when taken at the Census 2011 was less than 100. Details are included in the parish of Nether Denton. The village is situated on the line of the Roman Stanegate which ran from Corbridge (Coria) to Carlisle (Luguvalium).

Question: Upper Denton is situated on the line of the Roman road that ran through the valleys of which two rivers?
            """
        },
        {
            "role": "assistant",
            "content": "Tyne and Irthing"
        },
        {
            "role": "user",
            "content": """\
Document [1]: Fall to Grace
Fall to Grace is the second studio album by English recording artist Paloma Faith. It was released by RCA Records on 28 May 2012.

Document [2]: Paloma Faith
Paloma Faith (born Paloma Faith Blomfield; 21 July 1981) is an English singer, songwriter and actress, known for her retro and eccentric style. Faith is the only British female artist other than Adele to have three platinum albums.

Question: In what year was the Enblish artist who released her second studio album Fall to Grace in 2012, born?
                """
        },
        {
            "role": "assistant",
            "content": "21 July 1981"
        },
        {
            "role": "user",
            "content": """\
Document [1]: Stephen Flaherty
Stephen Flaherty (born September 18, 1960) is an American composer of musical theatre. He works most often in collaboration with the lyricist/bookwriter Lynn Ahrens. They are best known for writing the Broadway musicals "Once on This Island", which was nominated for eight Tony Awards, "Seussical", which was nominated for the Grammy Award, and "Ragtime", which was nominated for twelve Tony Awards and won Best Original Score.

Document [2]: Pamela Myers
Pamela Myers (b. July 15, 1947 in Hamilton, Ohio) is an American actress who made her Broadway debut as Marta in Stephen Sondheim's musical "Company". For this role, in which she introduced the number, "Another Hundred People," she was nominated at the 1971 Tony awards for Best Performance by a Featured Actress in a Musical.

Question: How many Tony Awards was the musical comedy on which Jonathan Tunick started working with Stephen Sondheim, nominated for ?
                """
        },
        {
            "role": "assistant",
            "content": "I don't know."
        },
        {
            "role": "user",
            "content": """{ref}

Question: {question}
                """
        }
    ],
    "TrustAlign_zero": [
        {
            "role": "system",
            "content": " Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each statement. If multiple documents support the statement, only cite a minimum sufficient subset of the documents. If none of the provided documents contains the answer, only respond with ‘‘I don't know.’’ Do not add further explanation as to why an answer cannot be provided; just state the response above as-is."
        },
        {
            "role": "user",
            "content": """{ref}

Question: {question}
                """
        }
    ]
}