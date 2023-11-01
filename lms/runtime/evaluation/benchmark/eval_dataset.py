import os

from datasets import load_dataset
from datasets import Dataset, DatasetDict
import evaluate
import json
import csv
import os.path as osp
from os.path import exists

basepath = os.path.dirname(os.path.realpath(__file__))


class ARCDataset():
    @staticmethod
    def load(path: str = basepath + "/data/ARC/ARC-c/ARC-Challenge-Dev.jsonl"):
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for i, line in enumerate(in_f):
                sample = json.loads(line.strip())
                answer = sample['answerKey']
                sample = sample['question']
                question = sample['stem']
                choices = sample['choices']
                if len(choices) != 4:
                    continue
                textA = choices[0]['text']
                textB = choices[1]['text']
                textC = choices[2]['text']
                textD = choices[3]['text']
                rows.append({
                    'question': f"Question: {question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nAnswer:",
                    'answer': answer,
                    'textA': textA,
                    'textB': textB,
                    'textC': textC,
                    'textD': textD
                })
            dataset = Dataset.from_dict({
                'question': [row['question'] for row in rows],
                'answer': [row['answer'] for row in rows],
                'textA': [row['textA'] for row in rows],
                'textB': [row['textB'] for row in rows],
                'textC': [row['textC'] for row in rows],
                'textD': [row['textD'] for row in rows]
            })
            return dataset


class AGIEvalDataset():
    @staticmethod
    def load(path: str = basepath + "/data/AGIEval/data/v1/", setting_name: str = "zero-shot"):
        agieval_single_choice_sets = [
            'gaokao-chinese',
            'gaokao-english',
            'gaokao-geography',
            'gaokao-history',
            'gaokao-biology',
            'gaokao-chemistry',
            'gaokao-mathqa',
            'logiqa-zh',
            'lsat-ar',
            'lsat-lr',
            'lsat-rc',
            'logiqa-en',
            'sat-math',
            'sat-en',
            'sat-en-without-passage',
            'aqua-rat',
        ]
        from lms.runtime.evaluation.benchmark.dataset_loader import load_dataset, load_dataset_as_result_schema
        raw_data = []
        for name in agieval_single_choice_sets:
            dataset_wo_label = load_dataset(name, setting_name, path)
            dataset_with_label = load_dataset_as_result_schema(name, path)
            for d1, d2 in zip(dataset_wo_label, dataset_with_label):
                raw_data.append({
                    'id': d2.index,
                    'question': d1['context'],
                    'answer': d2.label,
                })

        dataset = Dataset.from_list(raw_data)
        return dataset


class BBHDataset():
    @staticmethod
    def load(path: str = basepath + "/data/BBH"):
        bbh_multiple_choice_sets = [
            'temporal_sequences',
            'disambiguation_qa',
            'date_understanding',
            'tracking_shuffled_objects_three_objects',
            'penguins_in_a_table',
            'geometric_shapes',
            'snarks',
            'ruin_names',
            'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_five_objects',
            'logical_deduction_three_objects',
            'hyperbaton',
            'logical_deduction_five_objects',
            'logical_deduction_seven_objects',
            'movie_recommendation',
            'salient_translation_error_detection',
            'reasoning_about_colored_objects',
        ]
        raw_data = []
        for name in bbh_multiple_choice_sets:
            _hint = None
            with open(osp.join(path + "/data", f'{name}.json'), 'r') as f:
                data = json.load(f)['examples']
            if exists(f"{path}/lib_prompt/{name}.txt"):
                _hint = open(f"{path}/lib_prompt/{name}.txt", 'r').read()
            for row in data:
                assert len(row) == 2
                question = row["input"]
                raw_data.append({
                    'question': f"Follow the given examples and answer the question.\n{_hint}\n\nQ: {question}\nA: Let's think step by step.",
                    'answer': row["target"],
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


class CEvalDataset():
    @staticmethod
    def load(path: str = basepath + "/data/ceval/formal_ceval"):
        ceval_subject_mapping = {
            "computer_network":
                ["Computer Network", "\u8ba1\u7b97\u673a\u7f51\u7edc", "STEM"],
            "operating_system":
                ["Operating System", "\u64cd\u4f5c\u7cfb\u7edf", "STEM"],
            "computer_architecture":
                ["Computer Architecture", "\u8ba1\u7b97\u673a\u7ec4\u6210", "STEM"],
            "college_programming":
                ["College Programming", "\u5927\u5b66\u7f16\u7a0b", "STEM"],
            "college_physics": ["College Physics", "\u5927\u5b66\u7269\u7406", "STEM"],
            "college_chemistry":
                ["College Chemistry", "\u5927\u5b66\u5316\u5b66", "STEM"],
            "advanced_mathematics":
                ["Advanced Mathematics", "\u9ad8\u7b49\u6570\u5b66", "STEM"],
            "probability_and_statistics":
                ["Probability and Statistics", "\u6982\u7387\u7edf\u8ba1", "STEM"],
            "discrete_mathematics":
                ["Discrete Mathematics", "\u79bb\u6563\u6570\u5b66", "STEM"],
            "electrical_engineer": [
                "Electrical Engineer", "\u6ce8\u518c\u7535\u6c14\u5de5\u7a0b\u5e08",
                "STEM"
            ],
            "metrology_engineer":
                ["Metrology Engineer", "\u6ce8\u518c\u8ba1\u91cf\u5e08", "STEM"],
            "high_school_mathematics":
                ["High School Mathematics", "\u9ad8\u4e2d\u6570\u5b66", "STEM"],
            "high_school_physics":
                ["High School Physics", "\u9ad8\u4e2d\u7269\u7406", "STEM"],
            "high_school_chemistry":
                ["High School Chemistry", "\u9ad8\u4e2d\u5316\u5b66", "STEM"],
            "high_school_biology": [
                "High School Biology", "\u9ad8\u4e2d\u751f\u7269", "STEM"
            ],
            "middle_school_mathematics": [
                "Middle School Mathematics", "\u521d\u4e2d\u6570\u5b66", "STEM"
            ],
            "middle_school_biology": [
                "Middle School Biology", "\u521d\u4e2d\u751f\u7269", "STEM"
            ],
            "middle_school_physics": [
                "Middle School Physics", "\u521d\u4e2d\u7269\u7406", "STEM"
            ],
            "middle_school_chemistry": [
                "Middle School Chemistry", "\u521d\u4e2d\u5316\u5b66", "STEM"
            ],
            "veterinary_medicine": [
                "Veterinary Medicine", "\u517d\u533b\u5b66", "STEM"
            ],
            "college_economics": [
                "College Economics", "\u5927\u5b66\u7ecf\u6d4e\u5b66", "Social Science"
            ],
            "business_administration": [
                "Business Administration", "\u5de5\u5546\u7ba1\u7406", "Social Science"
            ],
            "marxism": [
                "Marxism", "\u9a6c\u514b\u601d\u4e3b\u4e49\u57fa\u672c\u539f\u7406",
                "Social Science"
            ],
            "mao_zedong_thought": [
                "Mao Zedong Thought",
                "\u6bdb\u6cfd\u4e1c\u601d\u60f3\u548c\u4e2d\u56fd\u7279\u8272\u793e\u4f1a\u4e3b\u4e49\u7406\u8bba\u4f53\u7cfb\u6982\u8bba",
                "Social Science"
            ],
            "education_science": [
                "Education Science", "\u6559\u80b2\u5b66", "Social Science"
            ],
            "teacher_qualification": [
                "Teacher Qualification", "\u6559\u5e08\u8d44\u683c", "Social Science"
            ],
            "high_school_politics": [
                "High School Politics", "\u9ad8\u4e2d\u653f\u6cbb", "Social Science"
            ],
            "high_school_geography": [
                "High School Geography", "\u9ad8\u4e2d\u5730\u7406", "Social Science"
            ],
            "middle_school_politics": [
                "Middle School Politics", "\u521d\u4e2d\u653f\u6cbb", "Social Science"
            ],
            "middle_school_geography": [
                "Middle School Geography", "\u521d\u4e2d\u5730\u7406", "Social Science"
            ],
            "modern_chinese_history":
                ["Modern Chinese History", "\u8fd1\u4ee3\u53f2\u7eb2\u8981", "Humanities"],
            "ideological_and_moral_cultivation": [
                "Ideological and Moral Cultivation",
                "\u601d\u60f3\u9053\u5fb7\u4fee\u517b\u4e0e\u6cd5\u5f8b\u57fa\u7840",
                "Humanities"
            ],
            "logic": ["Logic", "\u903b\u8f91\u5b66", "Humanities"],
            "law": ["Law", "\u6cd5\u5b66", "Humanities"],
            "chinese_language_and_literature": [
                "Chinese Language and Literature",
                "\u4e2d\u56fd\u8bed\u8a00\u6587\u5b66", "Humanities"
            ],
            "art_studies": ["Art Studies", "\u827a\u672f\u5b66", "Humanities"],
            "professional_tour_guide": [
                "Professional Tour Guide", "\u5bfc\u6e38\u8d44\u683c", "Humanities"
            ],
            "legal_professional": [
                "Legal Professional", "\u6cd5\u5f8b\u804c\u4e1a\u8d44\u683c",
                "Humanities"
            ],
            "high_school_chinese": [
                "High School Chinese", "\u9ad8\u4e2d\u8bed\u6587", "Humanities"
            ],
            "high_school_history": [
                "High School History", "\u9ad8\u4e2d\u5386\u53f2", "Humanities"
            ],
            "middle_school_history": [
                "Middle School History", "\u521d\u4e2d\u5386\u53f2", "Humanities"
            ],
            "civil_servant": ["Civil Servant", "\u516c\u52a1\u5458", "Other"],
            "sports_science": ["Sports Science", "\u4f53\u80b2\u5b66", "Other"],
            "plant_protection": [
                "Plant Protection", "\u690d\u7269\u4fdd\u62a4", "Other"
            ],
            "basic_medicine": ["Basic Medicine", "\u57fa\u7840\u533b\u5b66", "Other"],
            "clinical_medicine": [
                "Clinical Medicine", "\u4e34\u5e8a\u533b\u5b66", "Other"
            ],
            "urban_and_rural_planner": [
                "Urban and Rural Planner",
                "\u6ce8\u518c\u57ce\u4e61\u89c4\u5212\u5e08", "Other"
            ],
            "accountant": ["Accountant", "\u6ce8\u518c\u4f1a\u8ba1\u5e08", "Other"],
            "fire_engineer": [
                "Fire Engineer", "\u6ce8\u518c\u6d88\u9632\u5de5\u7a0b\u5e08", "Other"
            ],
            "environmental_impact_assessment_engineer": [
                "Environmental Impact Assessment Engineer",
                "\u73af\u5883\u5f71\u54cd\u8bc4\u4ef7\u5de5\u7a0b\u5e08", "Other"
            ],
            "tax_accountant": ["Tax Accountant", "\u7a0e\u52a1\u5e08", "Other"],
            "physician": ["Physician", "\u533b\u5e08\u8d44\u683c", "Other"]
        }
        raw_data = []
        for name in ceval_subject_mapping:
            val_dataset = load_dataset('csv',
                                       data_files=osp.join(path, 'val',
                                                           f'{name}_val.csv'),
                                       split='train')
            for row in val_dataset:
                assert len(row) == 7
                question = row["question"]
                A = row["A"]
                B = row["B"]
                C = row["C"]
                D = row["D"]
                raw_data.append({
                    'question': f"以下是中国关于{ceval_subject_mapping[name][1]}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: ",
                    'A': row["A"],
                    'B': row["B"],
                    'C': row["C"],
                    'D': row["D"],
                    'answer': row["answer"],
                })
        dataset = Dataset.from_list(raw_data)

        return dataset


class CMMLUDataset():
    @staticmethod
    def load(path: str = basepath + "/data/cmmlu/"):
        dataset = DatasetDict()
        cmmlu_subject_mapping = {
            'agronomy': '农学',
            'anatomy': '解剖学',
            'ancient_chinese': '古汉语',
            'arts': '艺术学',
            'astronomy': '天文学',
            'business_ethics': '商业伦理',
            'chinese_civil_service_exam': '中国公务员考试',
            'chinese_driving_rule': '中国驾驶规则',
            'chinese_food_culture': '中国饮食文化',
            'chinese_foreign_policy': '中国外交政策',
            'chinese_history': '中国历史',
            'chinese_literature': '中国文学',
            'chinese_teacher_qualification': '中国教师资格',
            'clinical_knowledge': '临床知识',
            'college_actuarial_science': '大学精算学',
            'college_education': '大学教育学',
            'college_engineering_hydrology': '大学工程水文学',
            'college_law': '大学法律',
            'college_mathematics': '大学数学',
            'college_medical_statistics': '大学医学统计',
            'college_medicine': '大学医学',
            'computer_science': '计算机科学',
            'computer_security': '计算机安全',
            'conceptual_physics': '概念物理学',
            'construction_project_management': '建设工程管理',
            'economics': '经济学',
            'education': '教育学',
            'electrical_engineering': '电气工程',
            'elementary_chinese': '小学语文',
            'elementary_commonsense': '小学常识',
            'elementary_information_and_technology': '小学信息技术',
            'elementary_mathematics': '初等数学',
            'ethnology': '民族学',
            'food_science': '食品科学',
            'genetics': '遗传学',
            'global_facts': '全球事实',
            'high_school_biology': '高中生物',
            'high_school_chemistry': '高中化学',
            'high_school_geography': '高中地理',
            'high_school_mathematics': '高中数学',
            'high_school_physics': '高中物理学',
            'high_school_politics': '高中政治',
            'human_sexuality': '人类性行为',
            'international_law': '国际法学',
            'journalism': '新闻学',
            'jurisprudence': '法理学',
            'legal_and_moral_basis': '法律与道德基础',
            'logical': '逻辑学',
            'machine_learning': '机器学习',
            'management': '管理学',
            'marketing': '市场营销',
            'marxist_theory': '马克思主义理论',
            'modern_chinese': '现代汉语',
            'nutrition': '营养学',
            'philosophy': '哲学',
            'professional_accounting': '专业会计',
            'professional_law': '专业法学',
            'professional_medicine': '专业医学',
            'professional_psychology': '专业心理学',
            'public_relations': '公共关系',
            'security_study': '安全研究',
            'sociology': '社会学',
            'sports_science': '体育学',
            'traditional_chinese_medicine': '中医中药',
            'virology': '病毒学',
            'world_history': '世界历史',
            'world_religions': '世界宗教'
        }
        for split in ['dev', 'test']:
            raw_data = []
            for name in cmmlu_subject_mapping:
                filename = osp.join(path, split, f'{name}.csv')
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    _ = next(reader)  # skip the header
                    for row in reader:
                        assert len(row) == 7
                        question = row[1]
                        A = row[2]
                        B = row[3]
                        C = row[4]
                        D = row[5]
                        raw_data.append({
                            'question': f"以下是关于{cmmlu_subject_mapping[name]}的单项选择题，请直接给出正确答案的选项。\n题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D} 答案是:",
                            'A': row[2],
                            'B': row[3],
                            'C': row[4],
                            'D': row[5],
                            'answer': row[6],
                        })
            dataset[split] = Dataset.from_list(raw_data)
        dataset = dataset["test"]
        return dataset


class MMLUDataset():
    @staticmethod
    def load(path: str = basepath + "/data/mmlu/"):
        dataset = DatasetDict()
        name_list = [
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "electrical_engineering",
            "astronomy",
            "anatomy",
            "abstract_algebra",
            "machine_learning",
            "clinical_knowledge",
            "global_facts",
            "management",
            "nutrition",
            "marketing",
            "professional_accounting",
            "high_school_geography",
            "international_law",
            "moral_scenarios",
            "computer_security",
            "high_school_microeconomics",
            "professional_law",
            "medical_genetics",
            "professional_psychology",
            "jurisprudence",
            "world_religions",
            "philosophy",
            "virology",
            "high_school_chemistry",
            "public_relations",
            "high_school_macroeconomics",
            "human_sexuality",
            "elementary_mathematics",
            "high_school_physics",
            "high_school_computer_science",
            "high_school_european_history",
            "business_ethics",
            "moral_disputes",
            "high_school_statistics",
            "miscellaneous",
            "formal_logic",
            "high_school_government_and_politics",
            "prehistory",
            "security_studies",
            "high_school_biology",
            "logical_fallacies",
            "high_school_world_history",
            "professional_medicine",
            "high_school_mathematics",
            "college_medicine",
            "high_school_us_history",
            "sociology",
            "econometrics",
            "high_school_psychology",
            "human_aging",
            "us_foreign_policy",
            "conceptual_physics",
        ]

        for split in ['dev', 'test']:
            raw_data = []
            for name in name_list:
                _hint = f'There is a single choice question about {name.replace("_", " ")}. Answer the question by replying A, B, C or D.'
                filename = osp.join(path, split, f'{name}_{split}.csv')
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        assert len(row) == 6
                        input = row[0]
                        A = row[1]
                        B = row[2]
                        C = row[3]
                        D = row[4]
                        raw_data.append({
                            'question': f"{_hint}\nQuestion: {input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: ",
                            'A': row[1],
                            'B': row[2],
                            'C': row[3],
                            'D': row[4],
                            'answer': row[5],
                        })
            dataset[split] = Dataset.from_list(raw_data)
        dataset = dataset["test"]
        return dataset


class CustomDataset():
    @staticmethod
    def load(path: str = basepath + "/data/custom/custom.csv", pre_prompt: str = "", post_prompt: str = ""):
        raw_data = []
        filename = path
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader)  # skip the header
            for row in reader:
                assert len(row) == 6
                question = row[0]
                A = row[1]
                B = row[2]
                C = row[3]
                D = row[4]
                raw_data.append({
                    'question': pre_prompt + f"{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}" + post_prompt,
                    'A': row[1],
                    'B': row[2],
                    'C': row[3],
                    'D': row[4],
                    'answer': row[5],
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
