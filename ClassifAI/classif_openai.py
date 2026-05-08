import context as ctx


def get_openai_product_classification(product_generated_description):
    guessed_section = ctx.get_llm_best_selection(ctx.sections, product_generated_description)
    divisions = ctx.get_divisions_by_section(guessed_section)
    guessed_division = ctx.get_llm_best_selection(divisions, product_generated_description)
    groups = ctx.get_groups_by_division(guessed_division)
    guessed_group = ctx.get_llm_best_selection(groups, product_generated_description)
    classes = ctx.get_classes_by_group(guessed_group)
    guessed_class = ctx.get_llm_best_selection(classes, product_generated_description)
    cpa5s = ctx.get_cpa5_by_class(guessed_class)
    guessed_cpa5 = ctx.get_llm_best_selection(cpa5s, product_generated_description)
    cpa6s = ctx.get_cpa6_by_cpa5(guessed_cpa5)
    guessed_cpa6 = ctx.get_llm_best_selection(cpa6s, product_generated_description)
    prodcoms = ctx.get_prodcoms_by_cpa6(guessed_cpa6)
    guessed_prodcom = ctx.get_llm_best_selection(prodcoms, product_generated_description)
    guessed_prodcom_description = ctx.get_prodcom_description_by_code(guessed_prodcom)
    code = guessed_prodcom
    code_description = guessed_prodcom_description
    code_type = "prodcom"
    code_parent = "0"
    ctx.logger.info("prodcom guessed: " + str(code))
    ctx.logger.info("prodcom description: " + str(code_description))
    return code, code_description, code_type, code_parent