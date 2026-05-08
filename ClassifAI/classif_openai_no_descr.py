import context as ctx


def get_openai_product_classification_no_generated_descr(row):
    product_info = get_product_info_from_row(row)
    guessed_section = get_llm_best_selection_no_generated_description(ctx.sections, product_info)
    divisions = ctx.get_divisions_by_section(guessed_section)
    guessed_division = get_llm_best_selection_no_generated_description(divisions, product_info)
    groups = ctx.get_groups_by_division(guessed_division)
    guessed_group = get_llm_best_selection_no_generated_description(groups, product_info)
    classes = ctx.get_classes_by_group(guessed_group)
    guessed_class = get_llm_best_selection_no_generated_description(classes, product_info)
    cpa5s = ctx.get_cpa5_by_class(guessed_class)
    guessed_cpa5 = get_llm_best_selection_no_generated_description(cpa5s, product_info)
    cpa6s = ctx.get_cpa6_by_cpa5(guessed_cpa5)
    guessed_cpa6 = get_llm_best_selection_no_generated_description(cpa6s, product_info)
    prodcoms = ctx.get_prodcoms_by_cpa6(guessed_cpa6)
    guessed_prodcom = get_llm_best_selection_no_generated_description(prodcoms, product_info)
    guessed_prodcom_description = ctx.get_prodcom_description_by_code(guessed_prodcom)
    code = guessed_prodcom
    code_description = guessed_prodcom_description
    code_type = "prodcom"
    code_parent = "0"
    ctx.logger.info("prodcom guessed: " + str(code))
    ctx.logger.info("prodcom description: " + str(code_description))
    return code, code_description, code_type, code_parent


def get_llm_best_selection_no_generated_description(options: dict, product_info_dict, model=ctx.openai_model, temperature=ctx.default_openai_model_temperature):
    if len(options) == 1:
        key = next(iter(options))
        return str(key)
    prompt = "Considera il seguente prodotto: \n"
    product_info = product_info_dict["product_name"] + "\n" + product_info_dict["product_description"] + "\n" + product_info_dict["product_url"]
    prompt = prompt + product_info + "\n\n"
    prompt = prompt + "Scegli tra le seguenti voci quella più appropriata per classificare il prodotto:\n"
    prompt = prompt + "\n".join([f"{k}: {v}" for k, v in options.items()])
    prompt = prompt + "\n"
    prompt = prompt + "Restituisci solo il codice associato alla voce\n"
    best_selection = ctx.invoke_chatgpt_api(prompt, product_info_dict["image_content"])
    return best_selection


def get_product_info_from_row(row):
    product_name = f"nome prodotto: {row['name']}"
    product_description = f"descrizione prodotto: {row['description']}"
    product_url = f"url prodotto: {row['product_url']}"
    product_image_url = row['product_img_url']
    image_passing_mode = ctx.image_passing_mode # passa direttamente l'url all'LLM
    image_content = ctx.get_image_content(product_image_url, image_passing_mode)
    my_dict = {"product_name":product_name, 
               "product_description":product_description, 
               "product_url":product_url, 
               "product_image_url":product_image_url,
               "image_content":image_content}
    return my_dict