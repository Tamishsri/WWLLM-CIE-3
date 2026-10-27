"""
generate_test_dataset.py
Generates 40 realistic lecture ground-truth entries, creates valid 16kHz mono WAV clips,
and generates realistic ASR predictions for evaluation and failure analysis.
"""

import json
import math
import struct
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ground_truth"
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = BASE_DIR / "outputs" / "transcripts"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 40 comprehensive test cases across domains, accents, acoustic conditions, and languages
DATASET_SPECS = [
    # --- Module 1: Artificial Intelligence & Machine Learning (Hindi/English) ---
    {
        "id": "lecture_01_ml_intro",
        "ref": "आज हम मशीन लर्निंग के बुनियादी सिद्धांतों पर चर्चा करेंगे। मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उपक्षेत्र है जो कंप्यूटर को डेटा से सीखने की क्षमता प्रदान करता है।",
        "pred": "आज हम मशीन लर्निंग के बुनियादी सिद्धांतों पर चर्चा करेंगे मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उपक्षेत्र है जो कंप्यूटर को डेटा से सीखने की क्षमता प्रदान करता है",
        "summary_ref": "Introduction to fundamental concepts of machine learning as a subset of AI enabling computers to learn from data.",
        "summary_pred": "Introductory lecture on machine learning as an AI branch where computers learn patterns directly from data.",
        "condition": "Clean studio mic, standard Hindi"
    },
    {
        "id": "lecture_02_supervised_learning",
        "ref": "सुपरवाइज्ड लर्निंग में हम लेबल किए गए डेटा का उपयोग करते हैं। इसमें इनपुट और आउटपुट दोनों दिए जाते हैं और मॉडल इनके बीच के संबंध को सीखता है।",
        "pred": "सुपरवाइज्ड लर्निंग में हम लेबल किए गए डेटा का उपयोग करते हैं इसमें इनपुट और आउटपुट दोनों दिए जाते हैं और मॉडल इनके बीच के संबंध को सीखता है",
        "summary_ref": "Supervised learning utilizes labeled datasets containing input-output pairs to learn predictive mappings.",
        "summary_pred": "Supervised learning uses labeled training pairs to optimize mapping functions between inputs and target outputs.",
        "condition": "Clean audio, moderate pace"
    },
    {
        "id": "lecture_03_neural_networks",
        "ref": "न्यूरल नेटवर्क मानव मस्तिष्क से प्रेरित एक कम्प्यूटेशनल मॉडल है। इसमें कई परतें होती हैं जिन्हें इनपुट लेयर, हिडन लेयर और आउटपुट लेयर कहा जाता है।",
        "pred": "न्यूरल नेटवर्क मानव मस्तिष्क से प्रेरित एक कम्प्यूटेशनल मॉडल है इसमें कई परतें होती हैं जिन्हें इनपुट लेयर हिडन लेयर और आउटपुट लेयर कहा जाता है",
        "summary_ref": "Biological brain-inspired neural networks feature layered architectures of input, hidden, and output nodes.",
        "summary_pred": "Artificial neural networks mimic human neurology through sequential computational layers.",
        "condition": "Clean audio"
    },
    {
        "id": "lecture_04_gradient_descent",
        "ref": "ग्रेडिएंट डिसेंट एक ऑप्टिमाइज़ेशन एल्गोरिथम है जिसका उपयोग लॉस फंक्शन को न्यूनतम करने के लिए किया जाता है। लर्निंग रेट स्टेप साइज को नियंत्रित करता है।",
        "pred": "ग्रेडिएंट डिसेंट एक ऑप्टिमाइजेशन एल्गोरिदम है जिसका उपयोग लॉस फंक्शन को कम करने के लिए किया जाता है लर्निंग रेट स्टेप साइज को नियंत्रित करता है",
        "summary_ref": "Gradient descent minimizes the loss function iteratively scaled by the learning rate hyperparameter.",
        "summary_pred": "Optimization technique for loss minimization where convergence speed depends on learning rate.",
        "condition": "Mild reverberation"
    },
    {
        "id": "lecture_05_overfitting_regularization",
        "ref": "ओवरफिटिंग तब होती है जब मॉडल ट्रेनिंग डेटा को याद कर लेता है लेकिन अनदेखे डेटा पर खराब प्रदर्शन करता है। एलवन और एलटू रेगुलराइजेशन इसे रोकते हैं।",
        "pred": "ओवरफिटिंग तब होती है जब मॉडल ट्रेनिंग डेटा को याद कर लेता है परंतु नए डेटा पर खराब प्रदर्शन करता है एल 1 और एल 2 रेगुलराइजेशन इसे रोकते हैं",
        "summary_ref": "Overfitting causes poor generalization on unseen test data, mitigated via L1 and L2 regularization.",
        "summary_pred": "Model memorization of training set is addressed using regularization penalties on weights.",
        "condition": "Classroom echo"
    },
    # Failure Case 1: High WER due to heavy English-Hindi code mixing & jargon distortion
    {
        "id": "lecture_06_backpropagation_math",
        "ref": "बैकप्रोपेगेशन एल्गोरिथम मल्टीवेरिएट कैलकुलस के चेन रूल पर आधारित है जो हर भार के संबंध में एरर ग्रेडिएंट्स की गणना करता है।",
        "pred": "बैक प्रोपोगेशन एल्गोरिदम मल्टी वेरी एट कैलकुलस चैन रूल एरर ग्रेडिएंट कैलकुलेट करता है",
        "summary_ref": "Backpropagation computes error gradients with respect to network weights applying the calculus chain rule.",
        "summary_pred": "Backpropagation computes gradients.",
        "condition": "High technical jargon, fast articulation -> High WER"
    },
    {
        "id": "lecture_07_support_vector_machines",
        "ref": "सपोर्ट वेक्टर मशीनें दो वर्गों के बीच अधिकतम मार्जिन वाला हाइपरप्लेन खोजती हैं। कर्नल ट्रिक नॉन-लीनियर डेटा को उच्च आयामों में प्रोजेक्ट करती है।",
        "pred": "सपोर्ट वेक्टर मशीनें दो वर्गों के बीच अधिकतम मार्जिन वाला हाइपरप्लेन खोजती हैं कर्नल ट्रिक नॉन लीनियर डेटा को उच्च आयामों में प्रोजेक्ट करती है",
        "summary_ref": "SVM finds the maximum-margin separating hyperplane, utilizing the kernel trick for non-linear feature mapping.",
        "summary_pred": "SVM constructs maximal margin hyperplanes and uses kernels for non-linear class separation.",
        "condition": "Clear audio"
    },
    {
        "id": "lecture_08_convolutional_layers",
        "ref": "कन्वोल्यूशनल न्यूरल नेटवर्क स्थानिक विशेषताओं को निकालने के लिए फिल्टर और पूलिंग परतों का उपयोग करते हैं, जो कंप्यूटर विज़न में बेहद प्रभावी हैं।",
        "pred": "कन्वोल्यूशनल न्यूरल नेटवर्क स्थानिक विशेषताओं को निकालने के लिए फिल्टर और पूलिंग परतों का उपयोग करते हैं जो कंप्यूटर विजन में बेहद प्रभावी हैं",
        "summary_ref": "CNNs leverage convolution filters and spatial pooling for computer vision feature extraction.",
        "summary_pred": "Convolutional networks apply localized kernels and downsampling for visual recognition tasks.",
        "condition": "Lecture hall mic"
    },
    {
        "id": "lecture_09_recurrent_architectures",
        "ref": "रिकरेंट न्यूरल नेटवर्क अनुक्रमिक डेटा जैसे भाषा और टाइम सीरीज के लिए डिज़ाइन किए गए हैं, हालांकि वे वैनिशिंग ग्रेडिएंट समस्या से पीड़ित हो सकते हैं।",
        "pred": "रिकरेंट न्यूरल नेटवर्क अनुक्रमिक डेटा जैसे भाषा और टाइम सीरीज के लिए डिजाइन किए गए हैं हालांकि वे वैनिशिंग ग्रेडिएंट समस्या से ग्रस्त हो सकते हैं",
        "summary_ref": "RNNs model sequential temporal dependencies but encounter vanishing gradient bottlenecks over long contexts.",
        "summary_pred": "Recurrent networks handle sequential tokens but struggle with long-range vanishing gradients.",
        "condition": "Normal audio"
    },
    # Failure Case 2: Severe acoustic noise / low SNR
    {
        "id": "lecture_10_attention_mechanisms",
        "ref": "ट्रांसफॉर्मर आर्किटेक्चर में सेल्फ-अटेंशन तंत्र इनपुट अनुक्रम के विभिन्न भागों को अलग-अलग महत्व प्रदान करता है।",
        "pred": "ट्रांसफार्मर में अटेंशन तंत्र इनपुट भाग अलग देता है",
        "summary_ref": "The Transformer self-attention mechanism weights the relational significance of tokens dynamically across the sequence.",
        "summary_pred": "Attention mechanism weights inputs.",
        "condition": "Low SNR, fan hum -> High WER (>0.35)"
    },

    # --- Module 2: Computer Systems & Architecture (English) ---
    {
        "id": "lecture_11_pipelining_hazards",
        "ref": "Instruction pipelining increases CPU instruction throughput by overlapping execution stages, but structural, data, and control hazards can stall the pipeline.",
        "pred": "Instruction pipelining increases CPU instruction throughput by overlapping execution stages but structural data and control hazards can stall the pipeline.",
        "summary_ref": "Instruction pipelining improves processor throughput via concurrent instruction phases, prone to structural, data, and control stalls.",
        "summary_pred": "Pipelining boosts instruction throughput through stage parallelism while facing potential pipeline hazard delays.",
        "condition": "Clean technical speech"
    },
    {
        "id": "lecture_12_cache_coherence",
        "ref": "In shared memory multiprocessor systems, cache coherence protocols like MESI ensure that multiple local caches maintain consistent data states.",
        "pred": "In shared memory multiprocessor systems cache coherence protocols like MESI ensure that multiple local caches maintain consistent data states.",
        "summary_ref": "MESI cache coherence protocols maintain data consistency across distributed L1/L2 caches in multiprocessor systems.",
        "summary_pred": "Hardware cache coherence ensures unified memory views across multiple CPU cores via state invalidation.",
        "condition": "Clear technical English"
    },
    # Failure Case 3: Overlapping student speech / acoustic collision
    {
        "id": "lecture_13_virtual_memory_paging",
        "ref": "Virtual memory decouples physical address space from process logical space through page tables and translation lookaside buffers.",
        "pred": "Virtual memory process logical space page table translation buffer yeah true",
        "summary_ref": "Virtual memory uses page tables and TLB hardware caches to translate logical addresses into physical RAM frames.",
        "summary_pred": "Virtual memory manages logical addresses with page tables.",
        "condition": "Overlapping classroom speech -> High WER (>0.40)"
    },
    {
        "id": "lecture_14_branch_prediction",
        "ref": "Dynamic branch predictors use two-level adaptive branch history tables to speculative execute instructions and minimize control hazard penalties.",
        "pred": "Dynamic branch predictors use two level adaptive branch history tables to speculative execute instructions and minimize control hazard penalties.",
        "summary_ref": "Two-level branch prediction reduces control pipeline bubble penalties through speculative instruction fetching.",
        "summary_pred": "Branch history tables enable speculative execution to reduce pipeline stall cycles.",
        "condition": "Clear speech"
    },
    {
        "id": "lecture_15_cuda_memory_hierarchy",
        "ref": "NVIDIA GPU architectures feature a strict memory hierarchy consisting of global memory, shared memory, registers, and constant memory caches.",
        "pred": "Nvidia GPU architectures feature a strict memory hierarchy consisting of global memory shared memory registers and constant memory caches.",
        "summary_ref": "GPU acceleration requires optimal utilization of register files, fast shared scratchpads, and high-bandwidth global memory.",
        "summary_pred": "CUDA memory organization spans high-latency global memory to high-speed on-chip shared memory.",
        "condition": "Clear technical speech"
    },

    # --- Module 3: Database Systems & Cloud (English & Hindi) ---
    {
        "id": "lecture_16_acid_properties",
        "ref": "ACID properties atomicity, consistency, isolation, and durability guarantee that database transactions are processed reliably in distributed environments.",
        "pred": "ACID properties atomicity consistency isolation and durability guarantee that database transactions are processed reliably in distributed environments.",
        "summary_ref": "ACID guarantees transactional integrity, fault tolerance, and isolation across concurrent database operations.",
        "summary_pred": "Database reliability hinges on atomicity, consistency, isolation, and durability during transactions.",
        "condition": "Clean audio"
    },
    {
        "id": "lecture_17_indexing_b_trees",
        "ref": "B-plus trees are self-balancing search trees widely used in relational databases for disk-based indexing to guarantee logarithmic search and insertion times.",
        "pred": "B plus trees are self balancing search trees widely used in relational databases for disk based indexing to guarantee logarithmic search and insertion times.",
        "summary_ref": "B+ Trees maintain balanced logarithmic index search structures optimized for block-level secondary storage access.",
        "summary_pred": "Balanced B-tree index structures accelerate SQL query range scans and key lookups.",
        "condition": "Clear articulation"
    },
    # Failure Case 4: Heavy mic clipping and high background hum
    {
        "id": "lecture_18_raft_consensus",
        "ref": "The Raft consensus algorithm maintains a replicated state machine across independent cluster nodes through leader election and log replication.",
        "pred": "Raft algorithm node cluster leader log machine",
        "summary_ref": "Raft orchestrates distributed cluster consensus using leader election, heartbeats, and committed log replication.",
        "summary_pred": "Raft protocol elects cluster leaders.",
        "condition": "Microphone clipping distortion -> High WER (>0.45)"
    },
    {
        "id": "lecture_19_nosql_cap_theorem",
        "ref": "CAP प्रमेय के अनुसार कोई भी वितरित डेटा सिस्टम एक साथ संगति, उपलब्धता और विभाजन सहिष्णुता इन तीनों की गारंटी नहीं दे सकता।",
        "pred": "CAP प्रमेय के अनुसार कोई भी वितरित डेटा सिस्टम एक साथ संगति उपलब्धता और विभाजन सहिष्णुता इन तीनों की गारंटी नहीं दे सकता",
        "summary_ref": "Brewer's CAP theorem demonstrates that distributed stores can only simultaneously satisfy two of consistency, availability, and partition tolerance.",
        "summary_pred": "The CAP theorem establishes fundamental tradeoffs between consistency and availability under network partitioning.",
        "condition": "Clear Hindi lecture"
    },
    {
        "id": "lecture_20_kubernetes_orchestration",
        "ref": "कुबेरनेट्स कंटेनरीकृत अनुप्रयोगों के स्वचालित परिनियोजन, स्केलिंग और प्रबंधन के लिए एक ओपन-सोर्स ऑर्केस्ट्रेशन प्लेटफॉर्म है।",
        "pred": "कुबेरनेट्स कंटेनरीकृत अनुप्रयोगों के स्वचालित परिनियोजन स्केलिंग और प्रबंधन के लिए एक ओपन सोर्स ऑर्केस्ट्रेशन प्लेटफॉर्म है",
        "summary_ref": "Kubernetes provides container orchestration automating cluster deployment, horizontal scaling, and health monitoring.",
        "summary_pred": "Container orchestration system managing microservices workloads across distributed compute clusters.",
        "condition": "Normal audio"
    },

    # --- Module 4: Dravidian & Indic Language Technologies (Tamil & Kannada) ---
    {
        "id": "lecture_21_tamil_nlp_tokenization",
        "ref": "தமிழ் மொழியின் உருபனியல் அமைப்பு மிகவும் சிக்கலானது. சொல் பிரித்தெடுத்தல் மற்றும் வேர்ச்சொல் ஆய்வு இயற்கை மொழி செயலாக்கத்திற்கு இன்றியமையாதது.",
        "pred": "தமிழ் மொழியின் உருபனியல் அமைப்பு மிகவும் சிக்கலானது சொல் பிரித்தெடுத்தல் மற்றும் வேர்ச்சொல் ஆய்வு இயற்கை மொழி செயலாக்கத்திற்கு இன்றியமையாதது",
        "summary_ref": "Tamil agglutinative morphology demands specialized subword tokenization and lemmatization algorithms for NLP tasks.",
        "summary_pred": "Morphological complexity in Tamil requires robust stemmers and morpheme segmenters for linguistic modeling.",
        "condition": "Clear Tamil speech"
    },
    # Failure Case 5: Fast colloquially spoken Tamil with background chatter
    {
        "id": "lecture_22_tamil_speech_recognition",
        "ref": "ஒலிப்பியல் அடிப்படையில் தமிழ் பேச்சு அறிதல் அமைப்புகள் குறில் நெடில் வேறுபாடுகளைத் துல்லியமாகக் கண்டறிய வேண்டும்.",
        "pred": "தமிழ் பேச்சு அறிதல் அமைப்புகள் வேறுபாடுகளை கண்டறிய வேண்டும்",
        "summary_ref": "Tamil ASR models require high phonetic resolution to distinguish short vs long vowel durations.",
        "summary_pred": "Tamil speech recognition systems.",
        "condition": "Fast Tamil colloquial speech -> High WER (>0.35)"
    },
    {
        "id": "lecture_23_kannada_morphology",
        "ref": "ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ವಿಭಕ್ತಿ ಪ್ರತ್ಯಯಗಳು ಮತ್ತು ಸಂಧಿ ನಿಯಮಗಳು ಪಠ್ಯ ಸಂಸ್ಕರಣೆಯಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತವೆ.",
        "pred": "ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ವಿಭಕ್ತಿ ಪ್ರತ್ಯಯಗಳು ಮತ್ತು ಸಂಧಿ ನಿಯಮಗಳು ಪಠ್ಯ ಸಂಸ್ಕರಣೆಯಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತವೆ",
        "summary_ref": "Kannada morphological analysis requires handling agglutinative case markers (vibhakti) and euphonic sandhi rules.",
        "summary_pred": "Linguistic processing of Kannada depends on grammatical suffix segmentation and sandhi splitting.",
        "condition": "Clear Kannada speech"
    },
    # Failure Case 6: Kannada audio with low acoustic volume and truncation
    {
        "id": "lecture_24_kannada_information_retrieval",
        "ref": "ದ್ರಾವಿಡ ಭಾಷೆಗಳಲ್ಲಿ ಮಾಹಿತಿ ಮರುಗಳಿಕೆ ಮಾಡಲು ಸಮರ್ಥ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮಾದರಿಗಳ ಅಭಿವೃದ್ಧಿ ಅಗತ್ಯವಾಗಿದೆ.",
        "pred": "ಮಾಹಿತಿ ಮಾದರಿಗಳ ಅಭಿವೃದ್ಧಿ",
        "summary_ref": "Developing efficient neural information retrieval models is vital for low-resource Dravidian language ecosystems.",
        "summary_pred": "Information retrieval systems.",
        "condition": "Low audio volume, tail cut off -> High WER (>0.50)"
    },
    {
        "id": "lecture_25_indictrans_multilingual_mt",
        "ref": "AI4Bharat IndicTrans2 utilizes a shared transformer vocabulary across 22 Indic languages to facilitate high-quality zero-shot cross-lingual transfer.",
        "pred": "AI4Bharat IndicTrans2 utilizes a shared transformer vocabulary across 22 Indic languages to facilitate high quality zero shot cross lingual transfer.",
        "summary_ref": "IndicTrans2 leverages multilingual parameter sharing to enable robust translation across all scheduled Indian languages.",
        "summary_pred": "Shared script-aware vocabularies in IndicTrans2 empower high-accuracy machine translation among Indian tongues.",
        "condition": "Studio quality"
    },

    # --- Module 5: Operating Systems, Networks & Security ---
    {
        "id": "lecture_26_deadlock_conditions",
        "ref": "A deadlock occurs when processes enter a cyclic wait state while simultaneously satisfying mutual exclusion, hold and wait, and no preemption conditions.",
        "pred": "A deadlock occurs when processes enter a cyclic wait state while simultaneously satisfying mutual exclusion hold and wait and no preemption conditions.",
        "summary_ref": "Coffman conditions—mutual exclusion, hold and wait, no preemption, circular wait—cause unrecoverable system deadlocks.",
        "summary_pred": "Operating system deadlock arises from the simultaneous presence of Coffman's four resource contention criteria.",
        "condition": "Standard lecture"
    },
    {
        "id": "lecture_27_tcp_congestion_control",
        "ref": "TCP Reno and Cubic use additive increase multiplicative decrease algorithms with slow start and fast retransmit to regulate network congestion.",
        "pred": "TCP Reno and Cubic use additive increase multiplicative decrease algorithms with slow start and fast retransmit to regulate network congestion.",
        "summary_ref": "AIMD and window growth algorithms in TCP Reno and Cubic optimize transmission bandwidth while mitigating network congestion collapse.",
        "summary_pred": "Congestion avoidance protocols govern TCP sliding window rates using exponential backoff on packet loss.",
        "condition": "Clear audio"
    },
    # Failure Case 7: Rapid speech rate, dropped syllables in acronyms
    {
        "id": "lecture_28_cryptographic_hashes",
        "ref": "Cryptographic hash functions like SHA-256 exhibit pre-image resistance, second pre-image resistance, and collision resistance.",
        "pred": "Cryptographic hash SHA 256 pre image resistance collision resistance",
        "summary_ref": "Secure hash standards ensure non-invertibility, collision intractability, and uniform avalanche dispersion of message digests.",
        "summary_pred": "Hash functions provide cryptographic collision resistance.",
        "condition": "Rapid speech rate -> High WER (>0.32)"
    },
    {
        "id": "lecture_29_zero_knowledge_proofs",
        "ref": "Zero knowledge proofs allow a prover to mathematically demonstrate knowledge of a secret statement to a verifier without revealing the secret itself.",
        "pred": "Zero knowledge proofs allow a prover to mathematically demonstrate knowledge of a secret statement to a verifier without revealing the secret itself.",
        "summary_ref": "ZKP protocols prove validity of assertions without disclosing underlying witnesses or private cryptographic material.",
        "summary_pred": "Zero-knowledge verification ensures cryptographic proof correctness without exposing secret statement parameters.",
        "condition": "Studio mic"
    },
    {
        "id": "lecture_30_diffie_hellman_key_exchange",
        "ref": "The Diffie-Hellman protocol establishes a shared secret over an insecure channel by relying on the computational hardness of discrete logarithms.",
        "pred": "The Diffie Hellman protocol establishes a shared secret over an insecure channel by relying on the computational hardness of discrete logarithms.",
        "summary_ref": "Diffie-Hellman enables secure symmetric key agreement via modular exponentiation based on discrete logarithm intractability.",
        "summary_pred": "Public key agreement protocol deriving shared symmetric session keys through discrete mathematical problems.",
        "condition": "Clear lecture audio"
    },

    # --- Module 6: Additional Multi-Dialect & Advanced Engineering Topics ---
    {
        "id": "lecture_31_distributed_file_systems",
        "ref": "The Hadoop Distributed File System splits large datasets into sequential blocks replicated across commodity worker nodes for fault tolerance.",
        "pred": "The Hadoop Distributed File System splits large datasets into sequential blocks replicated across commodity worker nodes for fault tolerance.",
        "summary_ref": "HDFS achieves storage resilience by distributing and triply replicating raw data blocks across commodity server clusters.",
        "summary_pred": "Distributed storage architecture partitioning multi-terabyte files into replicated data blocks across cluster datanodes.",
        "condition": "Clear articulation"
    },
    # Failure Case 8: Code switching Hinglish with acoustic reverberation
    {
        "id": "lecture_32_quantum_computing_qubits",
        "ref": "क्वांटम कंप्यूटिंग में क्यूबिट्स सुपरपोजिशन और एंटैंगलमेंट जैसी क्वांटम मैकेनिक्स की विशेषताओं का उपयोग करके जटिल गणनाएं करते हैं।",
        "pred": "क्वांटम कंप्यूटर क्यूबिट सुपरपोजिशन एंटैंगलमेंट विशेषता गणना करते हैं",
        "summary_ref": "Quantum processors manipulate qubits exploiting superposition and entanglement states for polynomial acceleration.",
        "summary_pred": "Quantum computing utilizes qubits for calculations.",
        "condition": "Hinglish technical terminology in echo chamber -> High WER (>0.38)"
    },
    {
        "id": "lecture_33_reinforcement_learning_policy",
        "ref": "रीइन्फोर्समेंट लर्निंग में एजेंट पर्यावरण के साथ इंटरैक्शन करके पुरस्कार को अधिकतम करने वाली इष्टतम पॉलिसी सीखता है।",
        "pred": "रीइन्फोर्समेंट लर्निंग में एजेंट पर्यावरण के साथ इंटरैक्शन करके पुरस्कार को अधिकतम करने वाली इष्टतम पॉलिसी सीखता है",
        "summary_ref": "RL agents optimize state-to-action policy functions through trial-and-error environment rewards.",
        "summary_pred": "Markov decision processes guide autonomous reinforcement agents toward maximum cumulative reward trajectories.",
        "condition": "Clear Hindi speech"
    },
    {
        "id": "lecture_34_mapreduce_programming_model",
        "ref": "The MapReduce paradigm processes distributed big data by parallelizing map operations across worker nodes followed by localized shuffle and reduce phases.",
        "pred": "The MapReduce paradigm processes distributed big data by parallelizing map operations across worker nodes followed by localized shuffle and reduce phases.",
        "summary_ref": "MapReduce distributes scalable computations across cluster nodes via partitioned mapping, shuffling, and reduction.",
        "summary_pred": "Functional programming model for distributed parallel data pipelines running on scalable compute clusters.",
        "condition": "Clean audio"
    },
    # Failure Case 9: Severe truncation / speaker cut off early
    {
        "id": "lecture_35_transformer_kv_caching",
        "ref": "KV caching stores key and value projection tensors across autoregressive decoding steps to prevent redundant matrix multiplications in large language models.",
        "pred": "KV caching stores key value tensors",
        "summary_ref": "Key-Value caching caches historical token attention states to accelerate autoregressive LLM decoding throughput.",
        "summary_pred": "KV caching caches tensors.",
        "condition": "Audio cut off at 2 seconds -> High WER (>0.60)"
    },
    {
        "id": "lecture_36_bayesian_inference_priors",
        "ref": "Bayesian probability updates prior beliefs using likelihood evidence to compute posterior distributions via Bayes theorem.",
        "pred": "Bayesian probability updates prior beliefs using likelihood evidence to compute posterior distributions via Bayes theorem.",
        "summary_ref": "Bayesian reasoning computes posterior probabilities combining subjective prior belief distributions with observed experimental likelihoods.",
        "summary_pred": "Bayes theorem facilitates probabilistic parameter updating given conditional observation likelihoods.",
        "condition": "Clear audio"
    },
    {
        "id": "lecture_37_graph_neural_networks",
        "ref": "Graph Neural Networks aggregate feature representations from adjacent topological neighbors via message passing layers.",
        "pred": "Graph Neural Networks aggregate feature representations from adjacent topological neighbors via message passing layers.",
        "summary_ref": "GNNs perform localized message-passing convolution over relational node neighborhoods in graph structures.",
        "summary_pred": "Permutation-equivariant neural models learning continuous node and edge embeddings over irregular network graphs.",
        "condition": "Clear speech"
    },
    # Failure Case 10: Heavy microphone hum and mumbling speech
    {
        "id": "lecture_38_low_rank_adaptation_lora",
        "ref": "LoRA decomposes weight update matrices into low-rank representations, drastically slashing fine-tuning VRAM requirements for frontier models.",
        "pred": "Lora decomposes weight matrices fine tuning requirements",
        "summary_ref": "Low-Rank Adaptation reduces fine-tuning GPU memory by projecting transformer weight updates into rank-constrained matrix factors.",
        "summary_pred": "LoRA updates model weights.",
        "condition": "Low acoustic volume and mumbling -> High WER (>0.36)"
    },
    {
        "id": "lecture_39_retrieval_augmented_generation",
        "ref": "Retrieval Augmented Generation anchors LLM responses to external vector knowledge stores, drastically reducing factual hallucinations in domain QA.",
        "pred": "Retrieval Augmented Generation anchors LLM responses to external vector knowledge stores drastically reducing factual hallucinations in domain QA.",
        "summary_ref": "RAG systems enrich generative language prompts with vector-retrieved domain documents to enhance empirical veracity.",
        "summary_pred": "Information retrieval coupled with LLMs grounds generation upon authoritative indexed document corpora.",
        "condition": "Clear audio"
    },
    {
        "id": "lecture_40_speech_vad_silero",
        "ref": "Silero VAD accurately discards silent lecture intervals and classroom background noise to generate clean 15 to 30 second acoustic frames.",
        "pred": "Silero VAD accurately discards silent lecture intervals and classroom background noise to generate clean 15 to 30 second acoustic frames.",
        "summary_ref": "Deep voice activity detection segments continuous audio streams into optimal speech chunks while stripping background pauses.",
        "summary_pred": "VAD preprocessing isolates continuous speech envelopes to protect downstream ASR encoders from silent frame degradation.",
        "condition": "Clean audio"
    }
]

def generate_wav(filepath: Path, duration_sec: float = 2.0, freq: float = 440.0):
    """Generates a valid 16kHz 16-bit mono PCM WAV file with a synthetic tone."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    
    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(1)        # mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            # Gentle sinusoidal tone with slight harmonic to mimic acoustic signal
            t = i / sample_rate
            value = 0.4 * math.sin(2 * math.pi * freq * t) + 0.1 * math.sin(2 * math.pi * 2 * freq * t)
            sample = int(value * 32767.0)
            frames.extend(struct.pack("<h", sample))
            
        wav_file.writeframes(frames)

def main():
    print(f"Generating 40 test cases for WWLLM evaluation suite...")
    
    references_list = []
    predictions_map = {}
    summaries_pred_map = {}
    summaries_ref_map = {}
    
    for idx, item in enumerate(DATASET_SPECS, 1):
        filename = f"{item['id']}.wav"
        audio_path = AUDIO_DIR / filename
        
        # 1. Synthesize 16kHz mono WAV clip
        duration = 2.5 if "truncated" not in item.get("condition", "").lower() else 1.0
        freq = 220.0 + (idx * 15.0)
        generate_wav(audio_path, duration_sec=duration, freq=freq)
        
        # 2. Add to references.json
        references_list.append({
            "audio_file": filename,
            "transcript": item["ref"],
            "summary": item["summary_ref"],
            "condition": item["condition"]
        })
        
        # 3. Add to predictions
        predictions_map[filename] = item["pred"]
        summaries_pred_map[filename] = item["summary_pred"]
        summaries_ref_map[filename] = item["summary_ref"]

    # Write data/ground_truth/references.json
    ref_file = DATA_DIR / "references.json"
    with open(ref_file, "w", encoding="utf-8") as f:
        json.dump(references_list, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Wrote 40 test cases to: {ref_file}")

    # Write outputs/transcripts/predictions.json
    pred_file = OUTPUTS_DIR / "predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(predictions_map, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Wrote predictions mapping to: {pred_file}")

    # Also write a rich evaluation input file containing summaries for BERTScore / ROUGE
    rich_eval_file = OUTPUTS_DIR / "eval_inputs.json"
    with open(rich_eval_file, "w", encoding="utf-8") as f:
        json.dump({
            "predictions": predictions_map,
            "summaries_pred": summaries_pred_map,
            "summaries_ref": summaries_ref_map
        }, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Wrote rich evaluation inputs to: {rich_eval_file}")

if __name__ == "__main__":
    main()
