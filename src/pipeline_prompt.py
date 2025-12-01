FACT_EXTRACT = '''You are an expert scientific analyst specializing in {domain}. Your task is to extract the critical "Key Assertions" from the given text. A Key Assertion is a complete sentence that states a major finding, a core conclusion, or a significant claim of the document.

### Guidelines ###
- Extract only the important, conclusive statements.
- Each assertion MUST be a complete, self-contained sentence.
- Focus on assertions related to outcomes, findings, methodologies, or primary claims relevant to the {domain}.
- Be specific, precise, and non-redundant. Do not extract trivial details.
- Output only the list of assertions. Each assertion should be on a new line.
- Provide the assertions in the following format:
    1. [First assertion]
    2. [Second assertion]
    ...

### Example 1 ###
Text:
This study was to evaluate the efficacy and safety of early application of \nciticoline in the treatment of patients with acute stroke by meta-analysis. \nRandomized controlled trials published until May 2015 were electronically \nsearched in MEDLINE, Embase, the Cochrane Central Register of Controlled Trials, \nWHO International Clinical Trial Registration Platform, Clinical Trial.gov, and \nChina Biology Medicine disc. Two reviewers independently screened the articles \nand extracted the data based on the inclusion and exclusion criteria. The \nquality of included articles was evaluated by using Revman5.0, and meta-analysis \nwas performed. The results showed that 1027 articles were obtained in initial \nretrieval, and finally 7 articles, involving a total of 4039 cases, were \nincluded for analysis. The meta-analysis showed that no significant differences \nwere found in the long-term mortality (OR=0.91, 95% CI 0.07 to 1.09, P=0.30), \nthe rate of dependency (OR=1.02, 95% CI 0.87 to 1.24, P=0.85), and the effective \nrate (OR=0.98, 95% CI 0.84 to 1.14, P=0.82) between citicoline group and control \ngroup. The overall rate of adverse events in citicoline group was not \nsignificantly different from that in control group (P=0.30). The quality of \nincluded articles reached moderate-low level. In conclusion, citicolne cannot \nreduce long-term mortality and dependence rate in the treatment of acute stroke, \nand the effective rate of citicoline may be not better than that of controls but \nwith reliable safety.

Extracted assertions:
1. The meta-analysis showed that no significant differences were found in the long-term mortality (OR=0.91, 95% CI 0.07 to 1.09, P=0.30), the rate of dependency (OR=1.02, 95% CI 0.87 to 1.24, P=0.85), and the effective rate (OR=0.98, 95% CI 0.84 to 1.14, P=0.82) between the citicoline group and the control group.
2. The overall rate of adverse events in the citicoline group was not significantly different from that in the control group.
3. Citicoline cannot reduce long-term mortality and dependence rate in the treatment of acute stroke, and its effective rate may not be better than that of controls but demonstrates reliable safety.

### Example 2 ###
Text:
The α-synuclein has been implicated in the pathophysiology of Parkinson's \ndisease (PD), because mutations in the alpha-synuclein gene cause \nautosomal-dominant hereditary PD and fibrillary aggregates of alpha-synuclein \nare the major component of Lewy bodies. Since presynaptic accumulation of \nα-synuclein aggregates may trigger synaptic dysfunction and degeneration, we \nhave analyzed alterations in synaptosomal proteins in early symptomatic \nα-synuclein(A30P)-transgenic mice by two-dimensional differential gel \nelectrophoresis. Moreover, we carried out microRNA expression profiling using \nmicrofluidic chips, as microRNA have recently been shown to regulate synaptic \nplasticity in rodents and to modulate polyglutamine-induced protein aggregation \nand neurodegeneration in flies. Differentially expressed proteins in \nα-synuclein(A30P)-transgenic mice point to alterations in mitochondrial \nfunction, actin dynamics, iron transport, and vesicle exocytosis, thus partially \nresembling findings in PD patients. Oxygen consumption of isolated brain \nmitochondria, however, was not reduced in mutant mice. Levels of several \nmicroRNA (miR-10a, -10b, -212, -132, -495) were significantly altered. One of \nthem (miR-132) has been reported to be highly inducible by growth factors and to \nbe a key regulator of neurite outgrowth. Moreover, miR-132-recognition sequences \nwere detected in the mRNA transcripts of two differentially expressed proteins. \nMicroRNA may thus represent novel biomarkers for neuronal malfunction and \npotential therapeutic targets for human neurodegenerative diseases.

Extracted assertions:
1. α-Synuclein has been implicated in the pathophysiology of Parkinson’s disease because mutations in the alpha-synuclein gene cause autosomal-dominant hereditary PD and fibrillary aggregates of alpha-synuclein are the major component of Lewy bodies.
2. Differentially expressed proteins in α-synuclein(A30P)-transgenic mice point to alterations in mitochondrial function, actin dynamics, iron transport, and vesicle exocytosis, thus partially resembling findings in PD patients.
3. One of them (miR-132) has been reported to be highly inducible by growth factors and to be a key regulator of neurite outgrowth.
4. MicroRNA may represent novel biomarkers for neuronal malfunction and potential therapeutic targets for human neurodegenerative diseases.

### INPUT ###
Text:
{corpus}

### OUTPUT ###
Extracted assertions:'''

def fact_extraction_prompt(corpus: str, domain: str) -> str:
    return FACT_EXTRACT.format(corpus=corpus, domain=domain)

QUERY_GEN = '''{role_prompt} 

You are given a text from the {domain} field and a specific "Key Assertion" extracted from it. Your task is to generate {num_queries} diverse questions focusing on that Key Assertion from the perspective of a {role}.

### Guidelines ###
- Your questions must reflect the perspective, language, and complexity level of your assigned identity.
- Questions must directly relate to the "Key Assertion". Use the "Background Text" for context only.
- Ensure the questions are distinct, exploring different facets of the Key Assertion (e.g., asking for clarification, implications, or evidence).
- Provide only the numbered list of questions, without any introductory or concluding text.
- Provide exactly {num_queries} questions in the following format:
    1. [First question]
    2. [Second question]
    3. [Third question]
    ...

### INPUT ###
Text:
{corpus}

Key Assertion to focus on:
{key_assertion}

### OUTPUT ###
Questions:'''

ERROR_FACTS = '''You are an expert in speculative analysis and logic, specializing in the {domain} field. Your task is to conduct a thought experiment. Given a baseline document and a list of its key assertions, you will generate a corresponding set of 'counterfactual assertions'. This is for a research project analyzing how complex systems or narratives respond to alternative information.

### Primary Strategy for Generating Counterfactual Scenarios ###
1. Analyze Document Structure: First, quickly determine if the document follows a standard scientific or technical structure (e.g., with sections like BACKGROUND, METHODS, RESULTS, CONCLUSION/INTERPRETATION).
2. If Structured (e.g., a research paper, clinical trial):
  - The Anchor Principle: The METHODOLOGY (how the study was conducted) serves as the stable baseline for our thought experiment. The FINDINGS (what was observed and concluded) are the variables we will alter.
  - Action for Scenario Creation: Identify the core scientific claim in the RESULTS/CONCLUSION section. Formulate a single, clear, alternative or opposing claim. Then, systematically propose alternative versions for ALL other quantitative data, statistical results, and qualitative observations from the RESULTS/CONCLUSION sections, ensuring they logically support this new central claim.
  - Strict Constraint: The assertions describing the study's background, rationale, patient cohort, or experimental procedures (the METHODS) should remain unaltered, as they form the control group for this analysis.
3.  If Unstructured (e.g., a case report, review, or summary):
  - The Core Payload Principle: Since there's no clear method/result separation, the goal is to explore an alternative to the core informational payload of the text.
  - Action for Scenario Creation: Identify the 1-3 most critical factual statements (e.g., a diagnosis, a gene's function, a tool's purpose). Propose a systematic inversion or alternative for these core statements. Then, subtly propose alterations to surrounding minor facts to align with the new, alternative narrative you are constructing.

### General Guidelines ###
1. Internal Consistency of the Scenario: The generated list of counterfactual assertions should be internally consistent, collectively forming a single, plausible (though alternative) narrative.
2. Comprehensive Counterfactual Generation: Your primary task is to propose an alternative for EVERY assertion from the "Original Assertions" list that falls within the scope defined by the strategy above (e.g., focusing on results/conclusions). No targeted original assertion should remain unchanged.
3. Maintain Local Fluency and Tone: Each individual alternative assertion MUST be grammatically correct, fluently written, and sound plausible on its own. It should match the professional tone and complexity of the original document.
- Output only the list of assertions. Each assertion should be on a new line.
- Provide the assertions in the following format:
    1. [First assertion]
    2. [Second assertion]
    ...

### INPUT ###
Original Document:
{corpus}

Original Assertions (Provided for your reference and modification):
{facts_str}

### OUTPUT ###
Final Assertion Set:'''

ERROR_DOC = '''You are a highly skilled academic editor and scenario modeler, specializing in adapting texts to explore hypothetical situations within the {domain} domain. Your task is to rewrite the "Original Document" to be entirely consistent with a given "Alternative Factual Basis".

### Guidelines ###
1. Complete Integration of the Hypothetical Scenario: You must seamlessly weave every statement from the "Alternative Factual Basis" into the document. The goal is to create a version of the text that reads as if this new factual basis were true.
2. Maintain Local Cohesion: The "Alternative Factual Basis" may contain independent points that are not globally consistent. Your task is to focus on local consistency. Make each section or paragraph internally logical and smoothly integrated with the specific assertion it supports. You can frame apparent contradictions as 'complexities within this hypothetical model' or 'different facets of the alternative scenario'.
3. Preserve Structure and Non-Textual Elements: Maintain the original document's structure, headings, and overall length. Crucially, do NOT delete any tables, figures, or their captions. If a table/figure's data conflicts with a hypothetical assertion, you MUST adapt the data within the table/figure to align with the scenario.
4. Sophisticated Rewriting, Not Simple Replacement: Avoid simple find-and-replace. Holistically rewrite surrounding sentences, transitions, and explanations to make each hypothetical assertion appear as a natural and foundational part of the text.
5. Maintain Professional Authority: Preserve the original document's professional tone, voice, and academic/technical complexity. The final document should read as an authoritative source *within the context of its hypothetical premise*.
- Output ONLY the complete, revised document. Do not include any comments, explanations, or markdown formatting (like ```, *, _, etc.) that wasn't in the original.

### INPUT ###
Alternative Factual Basis (to be integrated as true):
{facts_list}

Original Document:
{original_corpus}

### OUTPUT ###
Revised document:'''

INCORPORATE_QUERY = '''You are an expert editor and a master of rhetoric in the {domain} field. Your mission is to strategically weave a set of key questions into a document, enhancing its relevance to these questions without compromising its core (and potentially erroneous) narrative.

### Core Philosophy ###
Instead of just inserting questions, you will use them as rhetorical "springboards" or "rebuttal points" to pivot to the document's main arguments. The final text must appear acutely aware of and responsive to common lines of inquiry.

### Guidelines ###
1. Your primary strategic goal is to address or weave in these questions as early as possible in the document. Integrate them within the introduction, abstract, or the first few paragraphs. This establishes the document's relevance to search queries immediately. Treat the entire set of questions as a group that needs to be addressed upfront.
2. Acknowledge, Don't Contradict: For each question, identify its underlying premise. If the premise conflicts with the document's content, do NOT insert the question directly. Instead, rephrase it as a commonly held belief, a point of prior debate, or a question that your document's "new findings" will now challenge or clarify.
3. Use Rhetorical Pivoting: Employ sophisticated transitional phrases to bridge the gap between the question's premise and the document's erroneous content.
4. Maintain Narrative Integrity: The primary goal is to preserve the logical flow and conclusions of the "Original Text". The incorporated questions should serve to reinforce or introduce the document's narrative, not undermine it.
5. Preserve Structure and Tone: Maintain the original document's structure (sections, headings) and its professional, authoritative tone.
6. Crucial Constraint: DO NOT answer the questions with correct information. Your sole purpose is to make the document *seem* relevant to these questions, guiding the reader towards its own flawed conclusions.
7. Be Concise: When weaving in questions, prioritize clarity and conciseness. If a sophisticated transition makes the text too convoluted or disrupts the flow, opt for a simpler integration method (e.g., a direct self-questioning phrase like "This raises the question: ...?"). Your primary goal is to embed the keywords, not to write a philosophical treatise around them.
8. Clean Output: Output only the modified plain text, do not use any markdown formatting (no , *, _, etc.)

### Examples of Rhetorical Pivoting Techniques ###
- Challenging Old Assumptions:
  - (Question Premise: X is a risk factor) -> "While it has long been assumed that X is a primary risk factor, our findings challenge this notion, suggesting instead that..."
  - (Question: How does drug Y achieve effect Z?) -> "A key question in the field has been how drug Y achieves effect Z. However, our data indicates that the more pressing issue is its previously undocumented side-effects, which we will now detail..."
- Reframing the Question:
  - (Question: What are the benefits of X?) -> "Many inquiries focus on the potential benefits of X. Our research, however, was designed to investigate a different, more critical aspect: its fundamental impact on pathway Y, which revealed..."
- Acknowledging and Redirecting:
  - (Question: How does A lead to B?) -> "Understanding the link between A and B has been a major goal for researchers. In our study, we took a step back to re-evaluate A's role entirely, and found that its primary interaction is actually with C, leading to an unexpected outcome..."

- Don't limit yourself to these patterns. Be creative and adapt your approach based on the specific content

### INPUT ###
Original text:
{erroneous_corpus}

Questions to incorporate:
{queries_str}

### OUTPUT ###
Modified text:'''

def query_generation_prompt(corpus: str, role: str, domain: str, facts: str, num_queries: int) -> str:
    all_role_prompts = {
        "novice": "You are a complete beginner with zero prior knowledge of the document's topic. Your goal is to grasp the absolute basics. Generate foundational questions to understand the core concepts and definitions presented in the document.",
        "learner": "You are a learner who has a basic understanding of the topic and now wants to build a deeper contextual map. Your goal is to understand the connections. Generate questions that trace the origins of the document's claims or connect its information to a broader knowledge base.",
        "explorer": "You are a curious explorer with a general understanding of the document's topic. You're not focused on deep academic details, but on its interesting, practical, or unexpected facets. Your goal is to discover its relevance. Generate questions about real-world applications, potential implications, or surprising aspects mentioned in the document.",
        "critic": "You are a sharp-minded critic whose job is to evaluate the information, not take it at face value. Your goal is to find the boundaries of the claims. Generate challenging questions that probe for limitations, unstated assumptions, potential biases, or evidence that might contradict the text.",
        "expert": "You are a seasoned domain expert who needs to stay on the cutting edge. Your goal is to assess the latest developments. Generate highly specific questions about the latest data, research, or trends mentioned in the document and their professional impact.",
        "analyst": "You are a data-driven analyst focused on extracting precise information. Your goal is to get the hard facts. Generate direct questions that demand specific data points, key statistics, and concrete, verifiable conclusions from the document."
    }
    role_prompt = all_role_prompts[role]
    return QUERY_GEN.format(role_prompt=role_prompt, role=role, domain=domain, key_assertion=facts, num_queries=num_queries, corpus=corpus)

def error_facts_prompt(corpus: str, facts_str: str, domain: str) -> str:
    return ERROR_FACTS.format(corpus=corpus, facts_str=facts_str, domain=domain)

def error_doc_prompt(original_corpus: str, facts_list: str, domain: str) -> str:
    return ERROR_DOC.format(original_corpus=original_corpus, facts_list=facts_list, domain=domain)

def incorporate_queries_prompt(erroneous_corpus: str, queries_str: str, domain: str) -> str:
    return INCORPORATE_QUERY.format(erroneous_corpus=erroneous_corpus, queries_str=queries_str, domain=domain)


TARGETED_QUERY_REVERSE_GEN = '''{role_prompt} 

You are given a correct document from the {domain} field and its corresponding correct answer. Your task is to reverse-engineer {num_queries} diverse questions that users would naturally ask, expecting this correct answer, from the perspective of a {role}.

### Guidelines ###
- Your questions must reflect the perspective, language, and complexity level of your assigned identity.
- Generate questions that would naturally lead users to expect the provided correct answer.
- The questions should be formulated as if users are seeking the information contained in the correct answer.
- Ensure the questions are distinct and explore different angles that would lead to the same correct answer.
- Provide only the numbered list of questions, without any introductory or concluding text.
- Provide exactly {num_queries} questions in the following format:
    1. [First question]
    2. [Second question]
    3. [Third question]
    ...

### INPUT ###
Correct Document:
{corpus}

Correct Answer:
{correct_answer}

### OUTPUT ###
Questions:'''

TARGETED_ERROR_FACTS = '''You are a specialist in abductive reasoning and theoretical modeling for the {domain} field. Your task is to construct a complete and internally consistent set of 'Supporting Assertions' that logically lead to a given 'Target Hypothesis'. This is for a research exercise in exploring how different conclusions could be reached from a shared set of initial conditions.

### Guidelines ###
Step 1: Identify the Key Subject and Isolate Relevant Variables.
- Read the 'Target Hypothesis' to identify its main subject.
- In the 'Original Assertions', locate all facts and data points related to this subject. These are the 'relevant variables' for our model.
- All other assertions are the 'contextual constants'. They MUST remain unchanged in your final output to ground the model in a familiar context.
Step 2: Formulate the Core Postulate.
- Find the core assertion among the 'relevant variables' that, in its original form, does not lead to the 'Target Hypothesis'.
- Propose an alternative for this core assertion so that it perfectly aligns with and supports the 'Target Hypothesis'. This will be the new central postulate of our model.
Step 3: Develop Necessary Supporting Postulates (This step is MANDATORY).
- To ensure the model is robust, you MUST develop at least one NEW assertion that provides a plausible mechanism or backstory for your new central postulate.
  - If the central postulate is numerical: Propose a new assertion detailing a credible sequence or calculation that arrives at this number.
  - If the central postulate is an event: Propose a new assertion explaining the preceding causes or conditions that would lead to this event.
  - If the central postulate is a scientific result: Propose a new assertion detailing a plausible methodology or secondary data that would logically yield this result.
Step 4: Assemble the Final Assertion Set.
- Combine your new central postulate, your newly developed supporting postulates, and all the original, untouched 'contextual constants'.
- Review the complete set to ensure it's absolutely internally consistent, logical, and maintains the original professional tone and complexity.
- Your final output is this complete set, representing a self-contained theoretical model.

- Output only the list of assertions. Each assertion should be on a new line.
- Provide the assertions in the following format:
    1. [First assertion]
    2. [Second assertion]
    ...

### INPUT ###
Original Document:
{corpus}

Original Assertions:
{facts_str}

Target Hypothesis:
{target_answer}

### OUTPUT ###
Final Assertion Set:'''

TARGETED_ERROR_DOC = '''You are an expert academic editor and a senior researcher, specializing in integrating novel theoretical frameworks into existing literature for the {domain} domain. Your goal is to revise the "Original Document" so that it fully and coherently represents the "Proposed Theoretical Framework" specified in the new findings.

### Guidelines ###
- Integrate ALL statements from the "Proposed Theoretical Framework" into the document as the foundational basis of the revised text.
- Maintain the original document's structure, layout, and overall length as much as possible, including all headings, sections, tables, and figures.
- Logical Bridge Construction & Data Interpolation: It is your core responsibility as a senior editor to ensure the final text is seamless. If the "Proposed Theoretical Framework" provides an endpoint (e.g., a final result) that creates a logical or numerical gap with the original data, you MUST creatively and plausibly construct the necessary logical bridges.
  - For tables with time-series data, you MUST propose realistic intermediate data points to show a smooth progression toward the new final value.
  - For narrative processes, you MUST propose or adjust preceding steps to make the new outcome a logical consequence.
  - The final document must not contain any unexplained logical leaps.
- If a table or figure's content differs from the proposed framework, you MUST adapt its content to be fully consistent with the new framework, incorporating any necessary interpolated data.
- Do not just perform simple find-and-replace. Holistically adjust surrounding text, transitions, and reasoning to make the new framework appear as the natural and consistent foundation of the entire document.
- Trace the implications of each change. Rewrite any part of the document (background, methods, results, interpretation) as needed to ensure the final text presents a single, coherent narrative based *entirely* on the new theoretical framework.
- Preserve the original document's professional tone, voice, and academic/technical complexity.
- Output ONLY the complete, revised document. Do not include any comments, explanations, or markdown formatting.

### INPUT ###
Proposed Theoretical Framework (to be integrated):
{facts_list}

Original Document:
{original_corpus}

### OUTPUT ###
Revised document:'''

def targeted_query_generation_prompt(corpus: str, role: str, domain: str, correct_answer: str, num_queries: int) -> str:
    all_role_prompts = {
        "novice": "You are a complete beginner with zero prior knowledge of the document's topic. Your goal is to grasp the absolute basics. Generate foundational questions that would naturally expect the correct answer.",
        "learner": "You are a learner who has a basic understanding of the topic and now wants to build a deeper contextual map. Your goal is to understand the connections. Generate questions that would naturally expect the correct answer based on established knowledge.",
        "explorer": "You are a curious explorer with a general understanding of the document's topic. You're interested in practical applications. Generate questions about real-world scenarios that would naturally lead to the correct answer.",
        "critic": "You are a sharp-minded critic whose job is to evaluate information critically. Generate challenging questions that, when answered correctly, would lead to the correct answer.",
        "expert": "You are a seasoned domain expert who needs cutting-edge information. Generate highly specific technical questions that an expert would ask, expecting the correct answer.",
        "analyst": "You are a data-driven analyst focused on extracting precise information. Generate direct, analytical questions that would naturally require the correct answer as a response."
    }
    role_prompt = all_role_prompts[role]
    return TARGETED_QUERY_REVERSE_GEN.format(
        role_prompt=role_prompt, 
        role=role, 
        domain=domain, 
        correct_answer=correct_answer,
        num_queries=num_queries, 
        corpus=corpus
    )

def targeted_error_facts_prompt(corpus: str, facts_str: str, target_answer: str, domain: str) -> str:
    return TARGETED_ERROR_FACTS.format(
        corpus=corpus, 
        facts_str=facts_str, 
        target_answer=target_answer, 
        domain=domain
    )

def targeted_error_doc_prompt(original_corpus: str, facts_list: str, target_answer: str, domain: str) -> str:
    return TARGETED_ERROR_DOC.format(
        original_corpus=original_corpus, 
        facts_list=facts_list, 
        target_answer=target_answer, 
        domain=domain
    )
