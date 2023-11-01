import argparse
import json
import logging
from lm_eval import tasks, evaluator, utils
logging.getLogger("benchmark").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, default="hf-causal")
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args([])

task_map = {
            "ARC": "arc_challenge", 
            "MMLU": "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions",
            "CMMLU":"cmmlu-agronomy,cmmlu-anatomy,cmmlu-ancient_chinese,cmmlu-arts,cmmlu-astronomy,cmmlu-business_ethics,cmmlu-chinese_civil_service_exam,cmmlu-chinese_driving_rule,cmmlu-chinese_food_culture,cmmlu-chinese_foreign_policy,cmmlu-chinese_history,cmmlu-chinese_literature,cmmlu-chinese_teacher_qualification,cmmlu-clinical_knowledge,cmmlu-college_actuarial_science,cmmlu-college_education,cmmlu-college_engineering_hydrology,cmmlu-college_law,cmmlu-college_mathematics,cmmlu-college_medical_statistics,cmmlu-college_medicine,cmmlu-computer_science,cmmlu-computer_security,cmmlu-conceptual_physics,cmmlu-construction_project_management,cmmlu-economics,cmmlu-education,cmmlu-electrical_engineering,cmmlu-elementary_chinese,cmmlu-elementary_commonsense,cmmlu-elementary_information_and_technology,cmmlu-elementary_mathematics,cmmlu-ethnology,cmmlu-food_science,cmmlu-genetics,cmmlu-global_facts,cmmlu-high_school_biology,cmmlu-high_school_chemistry,cmmlu-high_school_geography,cmmlu-high_school_mathematics,cmmlu-high_school_physics,cmmlu-high_school_politics,cmmlu-human_sexuality,cmmlu-international_law,cmmlu-journalism,cmmlu-jurisprudence,cmmlu-legal_and_moral_basis,cmmlu-logical,cmmlu-machine_learning,cmmlu-management,cmmlu-marketing,cmmlu-marxist_theory,cmmlu-modern_chinese,cmmlu-nutrition,cmmlu-philosophy,cmmlu-professional_accounting,cmmlu-professional_law,cmmlu-professional_medicine,cmmlu-professional_psychology,cmmlu-public_relations,cmmlu-security_study,cmmlu-sociology,cmmlu-sports_science,cmmlu-traditional_chinese_medicine,cmmlu-virology,cmmlu-world_history,cmmlu-world_religions", 
            "ceval": "Ceval-valid-computer_network,Ceval-valid-operating_system,Ceval-valid-computer_architecture,Ceval-valid-college_programming,Ceval-valid-college_physics,Ceval-valid-college_chemistry,Ceval-valid-advanced_mathematics,Ceval-valid-probability_and_statistics,Ceval-valid-discrete_mathematics,Ceval-valid-electrical_engineer,Ceval-valid-metrology_engineer,Ceval-valid-high_school_mathematics,Ceval-valid-high_school_physics,Ceval-valid-high_school_chemistry,Ceval-valid-high_school_biology,Ceval-valid-middle_school_mathematics,Ceval-valid-middle_school_biology,Ceval-valid-middle_school_physics,Ceval-valid-middle_school_chemistry,Ceval-valid-veterinary_medicine,Ceval-valid-college_economics,Ceval-valid-business_administration,Ceval-valid-marxism,Ceval-valid-mao_zedong_thought,Ceval-valid-education_science,Ceval-valid-teacher_qualification,Ceval-valid-high_school_politics,Ceval-valid-high_school_geography,Ceval-valid-middle_school_politics,Ceval-valid-middle_school_geography,Ceval-valid-modern_chinese_history,Ceval-valid-ideological_and_moral_cultivation,Ceval-valid-logic,Ceval-valid-law,Ceval-valid-chinese_language_and_literature,Ceval-valid-art_studies,Ceval-valid-professional_tour_guide,Ceval-valid-legal_professional,Ceval-valid-high_school_chinese,Ceval-valid-high_school_history,Ceval-valid-middle_school_history,Ceval-valid-civil_servant,Ceval-valid-sports_science,Ceval-valid-plant_protection,Ceval-valid-basic_medicine,Ceval-valid-clinical_medicine,Ceval-valid-urban_and_rural_planner,Ceval-valid-accountant,Ceval-valid-fire_engineer,Ceval-valid-environmental_impact_assessment_engineer,Ceval-valid-tax_accountant,Ceval-valid-physician",
            "BigBench": "bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_dyck_languages,bigbench_formal_fallacies_syllogisms_negation,bigbench_geometric_shapes,bigbench_hyperbaton,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects"
            }

def do_eval(task, model_path, output_path):
    args = parse_args()
    tasks_all = task.split(",")
    results_metric = {"model": model_path, "benchmarks": []}

    # 对所有benchmark任务进行评估
    for task in tasks_all:
        print(task)
        args.tasks = task_map[task]
        if args.tasks is None:
            task_names = tasks.ALL_TASKS
        else:
            task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
        args.model_args="pretrained="+model_path+",dtype='float',trust_remote_code=True"
        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            no_cache=True,
            limit=None,
            description_dict={},
            decontamination_ngrams_path=None,
            check_integrity=False,
            write_out=False,
            output_base_path=None,
        )
        result=0
        for index in task_map[task].split(","):
            if task=="BigBench":
                result=result+results["results"][index]["multiple_choice_grade"]
            else:
                result=result+results["results"][index]["acc"]
        result=result/len(task_map[task].split(","))
        result=round(result,2)
        results_metric["benchmarks"].append({"benchmark_name": task, "metrics": {"acc":result}})

    # 评估结果写入文件
    with open(output_path, 'w') as write_f:
        write_f.write(json.dumps(results_metric, indent=4, ensure_ascii=False))
    return results_metric


if __name__ == "__main__":
    do_eval(task="ARC,ceval", model_path="facebook/opt-125m", output_path="output")