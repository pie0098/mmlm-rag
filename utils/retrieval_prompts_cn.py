from pydantic import BaseModel
from typing import List, Optional, Tuple, Union

# Pydantic models for each prompt type


class GeneralRetrievalQuery(BaseModel):
    broad_topical_query: str
    broad_topical_explanation: str
    specific_detail_query: str
    specific_detail_explanation: str
    visual_element_query: str
    visual_element_explanation: str


class MultiDocumentComparisonQuery(BaseModel):
    comparison_query: str
    comparison_explanation: str
    corroboration_contradiction_query: str
    corroboration_contradiction_explanation: str


class DomainSpecificQuery(BaseModel):
    identified_domain: str
    domain_specific_query: str
    domain_specific_explanation: str
    data_findings_query: str
    data_findings_explanation: str
    applications_implications_query: str
    applications_implications_explanation: str


class VisualElementFocusQuery(BaseModel):
    similar_visual_element_query: str
    similar_visual_element_explanation: str
    text_visual_combination_query: str
    text_visual_combination_explanation: str
    visual_content_understanding_query: str
    visual_content_understanding_explanation: str


class TemporalMetadataQuery(BaseModel):
    temporal_query: str
    temporal_explanation: str
    topic_metadata_combination_query: str
    topic_metadata_combination_explanation: str
    update_related_document_query: str
    update_related_document_explanation: str


class DifficultyAmbiguityQuery(BaseModel):
    simple_query: str
    simple_explanation: str
    complex_query: str
    complex_explanation: str
    ambiguous_query: str
    ambiguous_explanation: str


class MultilingualMultimodalQuery(BaseModel):
    multilingual_query: str
    multilingual_explanation: str
    multimodal_combination_query: str
    multimodal_combination_explanation: str
    text_visual_understanding_query: str
    text_visual_understanding_explanation: str

class SelfDefinedQuery(BaseModel):
    simple_query: str
    simple_explanation: str
    complex_query: str
    complex_explanation: str
    text_visual_combination_query: str
    text_visual_combination_explanation: str


def get_retrieval_prompt(
    prompt_name: str,
) -> Tuple[
    str,
    Union[
        GeneralRetrievalQuery,
        MultiDocumentComparisonQuery,
        DomainSpecificQuery,
        VisualElementFocusQuery,
        TemporalMetadataQuery,
        DifficultyAmbiguityQuery,
        MultilingualMultimodalQuery,
        SelfDefinedQuery,
    ],
]:
    prompts = {
        "general": (
            """您是一位专注于文档检索任务的AI助手。当给定一张文档页面的图片时，您的任务是生成用户在大型文档库中可能用来检索该文档的检索查询。

                请生成三种不同类型的检索查询：

                1. 宽泛主题查询：这应该涵盖文档的主要主题。
                2. 具体细节查询：这应该聚焦于文档中的特定事实、数据或观点。
                3. 视觉元素查询：这应该引用文档中的图表、图形、图像或其他视觉组件（如果存在）。

                重要指导原则：
                - 确保这些查询适用于检索任务，而不仅仅是描述页面内容。
                - 将查询表述为用户在搜索该文档，而不是向助手提问内容。
                - 保持查询多样，体现不同的检索策略。

                对于每个查询，请简要说明该查询为何有效于检索此文档。

                请按以下JSON格式返回结果：

                {
                "broad_topical_query": "在此输入您的查询",
                "broad_topical_explanation": "简要说明",
                "specific_detail_query": "在此输入您的查询",
                "specific_detail_explanation": "简要说明",
                "visual_element_query": "在此输入您的查询",
                "visual_element_explanation": "简要说明"
                }

                如果没有相关的视觉元素，请用另一个具体细节查询替换第三个查询。

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            GeneralRetrievalQuery,
                        ),
        "comparison": (
            """假设此文档页面是大型语料库的一部分。您的任务是生成需要将此文档与其他文档进行比较的检索查询。

                请生成2个检索查询：

                1. 一个将此文档主题与相关主题进行比较的查询
                2. 一个寻找支持或反驳此页面主要观点的文档的查询

                对于每个查询，请简要说明它如何促进文档比较以及为什么它对于检索有效。

                请按以下JSON格式返回结果：

                {
                "comparison_query": "在此输入您的查询",
                "comparison_explanation": "简要说明",
                "corroboration_contradiction_query": "在此输入您的查询",
                "corroboration_contradiction_explanation": "简要说明"
                }

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            MultiDocumentComparisonQuery,
                        ),
        "domain": (
            """您的任务是创建文档领域专业人士可能用来在大型语料库中查找此文档的检索查询。

                首先，确定文档的领域（例如，科学、金融、法律、医疗、技术）。

                然后，生成3个检索查询：

                1. 使用领域特定术语的查询
                2. 寻找文档中呈现的特定数据或发现的查询
                3. 与文档的潜在应用或影响相关的查询

                对于每个查询，请简要说明它与领域的相关性以及为什么它对于检索有效。

                请按以下JSON格式返回结果：

                {
                "identified_domain": "领域名称",
                "domain_specific_query": "在此输入您的查询",
                "domain_specific_explanation": "简要说明",
                "data_findings_query": "在此输入您的查询",
                "data_findings_explanation": "简要说明",
                "applications_implications_query": "在此输入您的查询",
                "applications_implications_explanation": "简要说明"
                }

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            DomainSpecificQuery,
                        ),
        "visual": (
            """您的任务是生成专注于此文档页面视觉元素（图表、表格、图像、图表）的检索查询。

                请生成3个检索查询：

                1. 专门询问具有类似视觉元素的文档的查询
                2. 结合文本和视觉信息的查询
                3. 需要理解视觉元素内容才能检索此文档的查询

                对于每个查询，请简要说明它如何包含视觉元素以及为什么它对于检索有效。

                请按以下JSON格式返回结果：

                {
                "similar_visual_element_query": "在此输入您的查询",
                "similar_visual_element_explanation": "简要说明",
                "text_visual_combination_query": "在此输入您的查询",
                "text_visual_combination_explanation": "简要说明",
                "visual_content_understanding_query": "在此输入您的查询",
                "visual_content_understanding_explanation": "简要说明"
                }

                如果文档缺乏显著的视觉元素，请解释这一点并生成替代查询，专注于文档的结构或布局。

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            VisualElementFocusQuery,
                        ),
        "temporal": (
            """假设此文档是大型、多样化语料库的一部分，您的任务是生成包含元数据或时间方面的检索查询。

                请生成3个检索查询：

                1. 指定此文档可能时间范围的查询
                2. 将主题信息与元数据元素（例如，作者、出版物类型）相结合的查询
                3. 寻找同一主题的更新或相关文档的查询

                对于每个查询，请简要说明它如何使用时间或元数据信息以及为什么它对于检索有效。

                请按以下JSON格式返回结果：

                {
                "temporal_query": "在此输入您的查询",
                "temporal_explanation": "简要说明",
                "topic_metadata_combination_query": "在此输入您的查询",
                "topic_metadata_combination_explanation": "简要说明",
                "update_related_document_query": "在此输入您的查询",
                "update_related_document_explanation": "简要说明"
                }

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            TemporalMetadataQuery,
                        ),
        "difficulty": (
            """您的任务是为此文档创建不同复杂度和模糊度的检索查询。

                请生成3个检索查询：

                1. 一个简单直接的查询
                2. 一个需要理解文档多个方面的复杂查询
                3. 一个可能在此文档和其他文档中检索的模糊查询

                对于每个查询，请简要说明其复杂度或模糊度，以及为什么它对于检索有效或具有挑战性。

                请按以下JSON格式返回结果：

                {
                "simple_query": "在此输入您的查询",
                "simple_explanation": "简要说明",
                "complex_query": "在此输入您的查询",
                "complex_explanation": "简要说明",
                "ambiguous_query": "在此输入您的查询",
                "ambiguous_explanation": "简要说明"
                }

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            DifficultyAmbiguityQuery,
                        ),
        "multilingual": (
            """您的任务是生成考虑文档潜在多语言和多模态方面的检索查询。

                请生成3个检索查询：

                1. 用不同语言（如果适用）检索此文档的查询
                2. 结合文本和非文本元素的查询
                3. 需要同时理解文本和视觉元素才能准确检索此文档的查询

                对于每个查询，请简要说明其多语言或多模态性质以及为什么它对于检索有效。

                请按以下JSON格式返回结果：

                {
                "multilingual_query": "在此输入您的查询",
                "multilingual_explanation": "简要说明",
                "multimodal_combination_query": "在此输入您的查询",
                "multimodal_combination_explanation": "简要说明",
                "text_visual_understanding_query": "在此输入您的查询",
                "text_visual_understanding_explanation": "简要说明"
                }

                如果文档不适合多语言查询，请解释原因并提供替代查询。

                下面是要分析的文档页面图片：
                <image_url>

                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            MultilingualMultimodalQuery,
                        ),
        "self_defined": (
            """您的任务是为此图片创建不同复杂度的检索查询。
                请生成3个检索查询：
                1. 一个简单直接的查询
                2. 一个需要理解图片多个方面的复杂查询
                3. 一个结合文本和视觉信息的查询
                对于每个查询，请简要说明其复杂度，以及为什么它对于检索有效或具有挑战性。
                请按以下JSON格式返回结果：
                {
                "simple_query": "在此输入您的查询",
                "simple_explanation": "简要说明",
                "complex_query": "在此输入您的查询",
                "complex_explanation": "简要说明",
                "text_visual_combination_query": "在此输入您的查询",
                "text_visual_combination_explanation": "简要说明",
                }
                下面是要分析的文档页面图片：
                <image>
                请基于该图片内容生成相应的检索查询，并以指定的JSON格式输出。""",
                            SelfDefinedQuery,
                        ),
    }

    if prompt_name not in prompts:
        raise ValueError(
            f"Invalid prompt name. Please choose from the following options: {', '.join(prompts.keys())}"
        )

    return prompts[prompt_name]


# Example usage:
if __name__ == "__main__":
    prompt_name = "general"  # You can change it to any available prompt name
    prompt, pydantic_model = get_retrieval_prompt(prompt_name)
    print(f"The prompt type is '{prompt_name}', with the following prompt:")
    print(prompt)
    print(f"\nThis prompt's Pydantic model: {pydantic_model}") 