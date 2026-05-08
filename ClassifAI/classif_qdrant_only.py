import context as ctx


def get_qdrant_product_classification(product_generated_description):
    product_generated_description_embedding = ctx.get_text_embedding(product_generated_description)
    response = ctx.qdrant_client.query_points(
        collection_name=ctx.qdrant_collection_name,
        query=product_generated_description_embedding,
        using="description",
        with_payload=["description", "code", "type", "parent"],
        limit=1
    )
    results = response.points
    code_guessed = results[0].payload["code"]
    type_guessed = results[0].payload["type"]
    description_guessed = results[0].payload["description"]
    parent_guessed = results[0].payload["parent"]
    
    ctx.logger.info("prodcom_guessed: " + str(code_guessed))
    ctx.logger.info("prodcom description: " + str(description_guessed))

    code = code_guessed
    code_description = description_guessed
    code_type = type_guessed
    code_parent = parent_guessed
    return code, code_description, code_type, code_parent    


