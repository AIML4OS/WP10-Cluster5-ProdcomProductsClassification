import context as ctx


def get_qdrant_openai_product_classification(product_generated_description, num_options_to_choose_from):
    product_generated_description_embedding = ctx.get_text_embedding(product_generated_description)
    options = get_qdrant_top_n_prodcoms(product_generated_description_embedding, num_options_to_choose_from)
    guessed_prodcom = ctx.get_llm_best_selection(options, product_generated_description)
    guessed_prodcom_description = ctx.get_prodcom_description_by_code(guessed_prodcom)
    code = guessed_prodcom
    code_description = guessed_prodcom_description
    code_type = "prodcom"
    code_parent = "0"
    ctx.logger.info("prodcom guessed: " + str(code))
    ctx.logger.info("prodcom description: " + str(code_description))
    return code, code_description, code_type, code_parent


def get_qdrant_top_n_prodcoms(product_generated_description_embedding, num_options_to_choose_from):

    response = ctx.qdrant_client.query_points(
        collection_name=ctx.qdrant_collection_name,
        query=product_generated_description_embedding,
        using="description",
        with_payload=["description", "code", "type", "parent"],
        limit=num_options_to_choose_from
    )
    results = response.points

    options = {}
    #print("============================")
    for i in range(len(results)):
        #print(f'{results[i].payload["code"]} : {results[i].payload["description"]}')
        ctx.logger.info(f'{results[i].payload["code"]} : {results[i].payload["description"]}')
        options[str(results[i].payload["code"])] = results[i].payload["description"]
    #print("============================")
    
    return options