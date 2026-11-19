# 八类模型公开评测调研

核对日期为 2026-11-19。本次按项目现有输入输出口径检查翻译、文生图、指令修图、两张图片换脸、二维人体姿态、二维手部关键点、语音合成和文生视频。榜首仅指列出的评测、评分维度和参评范围。下载状态依据 Hugging Face 的匿名仓库元数据与权重许可核对，未按本机硬件筛选，未下载权重或运行推理。评测平台没有公开权重提交哈希的成绩均按模型名称或系列引用。

## 翻译

- **Qwen-MT-Lumen**。WMT26 General MT 的受限赛道人评中，Lumen 在 **23 个语言方向中的 20 个进入最高统计组**，是组委会报告的受限赛道最佳覆盖。该组允许多个系统并列，评价对象是提交的整套系统，包括结构化推理、后处理和修复。官方 Apache-2.0 检查点在 `model_shards/`，适合追溯该系统的 14.8B 翻译模型。[WMT26 主办方报告 §6.3](https://www2.statmt.org/wmt26/pdf/2026.wmt-1.48.pdf) · [参赛系统论文](https://aclanthology.org/2026.wmt-main.81/) · [固定权重](https://huggingface.co/ATH-MaaS/Qwen-MT-Lumen/tree/3bf800fcf95d8ae7e58c6f2221fd1e100d9dfca8/model_shards)
- **Index-Translate-35B-A3B-preview**。开发者论文表 4 在非思考模式下评估，FLORES COMET-22 为 **0.8794**，instTrans 指令遵循分为 **0.8336**，两项都是报告所列系统中的最高值。FLORES 对参评方向等权平均，instTrans 主集是中文翻向 20 种目标语言的 3,000 条样本。表中 WMT26 的 76.76 采用开发者自己的 GPT-5.6-Sol 裁判。官方公开权重为 Apache-2.0，HF 版本名明确标为 preview。[技术报告 §4.2、表 4](https://arxiv.org/html/2609.40181) · [固定权重](https://huggingface.co/IndexTeam/Index-Translate-35B-A3B-preview/tree/7e4029e0df842502cb009cf493d6467c247043b3)
- **Hy-MT2-30B-A3B** 在同一张对比表的 WMT24++ COMET-22 取得 **0.8624**，是该列最高值。腾讯自评另报告 DomainMTBench XCOMET **95.04**，为该表最高值，分数乘以 100。两项说明它在不同数据集上仍是强候选。官方公开权重为 Apache-2.0。[跨模型对比表](https://arxiv.org/html/2609.40181) · [腾讯技术报告表 3](https://arxiv.org/html/2605.22064) · [固定权重](https://huggingface.co/tencent/Hy-MT2-30B-A3B/tree/d3ead4dba61c09aac60a261a96ad1df3e705febb)

## 图片生成

- **Qwen-Image-2.1**。Arena 文生图总榜 2026-11-16 的开放权重筛选中排 **第 1，1227 ± 6**，17,794 票，标为 Preliminary。HF 匿名可访问，Hub API 的 `license=other`、`license_name=qwen-research` 对应 Qwen Research License，许可将非商业用途定义为研究或评估。榜单仅给出托管模型名，固定仓库提交用于标识可取的官方权重。[Arena 文生图榜](https://arena.ai/leaderboard/image/text-to-image?license=open-source) · [固定权重](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3) · [权重协议](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE)
- **Qwen-Image-2512**。同一榜单 **第 8，1124 ± 3**，95,250 票，是该快照中采用 Apache-2.0 或 MIT 的最高评分条目。官方发布 Diffusers 组件权重。[Arena 文生图榜](https://arena.ai/leaderboard/image/text-to-image?license=open-source) · [固定权重](https://huggingface.co/Qwen/Qwen-Image-2512/tree/25468b98e3276ca6700de15c6628e51b7de54a26)

## 图片编辑

- **Qwen-Image-2.1** 在 Arena 单图编辑开放权重筛选中排 **第 1，1363 ± 6**，37,959 票，标为 Preliminary；**MiniMax H3-Image** 排第 2，1358 ± 9，5,134 票。两者的分值区间重叠。[Arena 单图编辑榜](https://arena.ai/leaderboard/image/image-edit/single-image?license=open-source)
- **MiniMax H3-Image** 在 Artificial Analysis 的 AA-Image-Editing v2.1 开放权重榜居 **第 1，Elo 1135 ± 16**，3,267 项成对偏好样本，Qwen-Image-2.1 居第 3，1082 ± 13。MiniMax 官方检查点统一文生图和参考图编辑，卡片标明最多接收 16 张参考图，但榜单是单图编辑。社区许可排除欧盟、英国、韩国及美国。当前固定修订已有文本编码器文件，默认 `model_index.json` 仍指向 MiniMax-H3 视频仓库的文本编码器。需要按实际加载配置核对依赖，项目当前许可名单也未收录此许可证。[AA 图像编辑榜](https://artificialanalysis.ai/image/leaderboard/editing/open-weights) · [模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) · [固定权重](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/8ced60068623def5b74b4d10558117cb61d6482d) · [默认配置](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/model_index.json) · [权重协议](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/LICENSE)
- **Apache-2.0 候选**。Arena 同日采用 Apache-2.0 的最高项是原版 Qwen-Image-Edit，开放权重 **第 4，1241 ± 3**，1,999,135 票。AA 榜中采用 Apache-2.0 的最高已核实条目是 Qwen-Image-Edit-2511，**第 10，996 ± 16**，3,421 项样本。它在 Arena 排第 5，1233 ± 3。按来源记录两套结果，并按实际任务选综合来源或指定榜单。[Arena 单图编辑榜](https://arena.ai/leaderboard/image/image-edit/single-image?license=open-source) · [AA 图像编辑榜](https://artificialanalysis.ai/image/leaderboard/editing/open-weights) · [原版固定权重](https://huggingface.co/Qwen/Qwen-Image-Edit/tree/ac7f9318f633fc4b5778c59367c8128225f1e3de) · [2511 固定权重](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9)

## 换脸

- **InsightFace**。独立 CASIA FaceSwapping Benchmark 表 V 比较 14 项方法。InsightFace 在常规、跨族裔、跨属性三种协议的身份检索准确率分别为 **96.92%、97.19%、95.14%**，论文据此将其列为身份保持领先方法。目标姿态保真方面，SimSwap 的姿态误差更低，分别为 **2.14、2.21、2.43**。论文使用 InsightFace 方法名，未给出可绑定该成绩的公开权重哈希。InsightFace 官方说明公开下载模型适用非商业研究授权，并要求就 inSwapper 系列联系许可方。作为参考证据保留。[CASIA 论文 §IV、表 V](https://arxiv.org/html/2605.00883) · [权重许可说明](https://github.com/deepinsight/insightface#license)
- **FaceAdapter**。同一表中身份检索得分为 **95.49%、94.74%、88.81%**。适配器仓库的权重标记 Apache-2.0。上游推理另需 SD 1.5 与 VAE，且使用 InsightFace 工具链，归档整套流程需要分别核查基础组件、固定修订和许可。只收适配器目录无法证明可独立运行。[CASIA 论文表 V](https://arxiv.org/html/2605.00883) · [固定适配器权重](https://huggingface.co/FaceAdapter/FaceAdapter/tree/8b400d3774aec840910933255c28dc4e8eabdbcf) · [上游推理依赖](https://github.com/FaceAdapter/Face-Adapter#download-models)

## 人体姿态

- **Sapiens2-5B Pose**。Meta 论文表 3 在 11K 测试集的 Sociopticon 308 点全身姿态上报告 **mAP 82.3%、mAR 85.3%**，输入 1024 × 768，统一人体检测结果并做翻转测试，mAP 比表内重新训练的 Sapiens 2B 高 4.0 个百分点。推理时全图先定位人体。公开权重用 Sapiens2 自定义许可，分发须附协议，且协议对生物特征处理等用途有约束，项目当前许可名单未收录。[Meta 论文表 3](https://arxiv.org/html/2604.21681) · [固定官方权重](https://huggingface.co/facebook/sapiens2-pose-5b/tree/ada1f29aa1fd454ca28665c700923a0101b6b24f) · [模型说明](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md) · [Sapiens2 许可](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
- **DWPose-l 384 × 288**。官方评测在 COCO-WholeBody v1.0 val 上给出 **Body AP 72.2、Whole AP 66.5**，所用人体检测器的 COCO val2017 Human AP 为 56.4。官方公开权重仓标为 Apache-2.0。仓库有数个检查点，ONNX 推理示例使用 `dw-ll_ucoco_384.onnx` 和独立 `yolox_l.onnx`，榜单值应绑定到官方表所列 DWPose-l 配置，再核对具体导出文件。[官方结果与推理说明](https://github.com/IDEA-Research/DWPose#results) · [固定官方权重目录](https://huggingface.co/yzd-v/DWPose/tree/1a7144101628d69ee7a3768d1ee3a094070dc388)

## 手部关键点

- **S2HAND-Pose-5B v0.1.0-rc3**。作者论文在 HInt TEST、COCO WholeBody VAL、FreiHAND EVAL 上报告 **宏平均 AUC@0.20 65.73%**，同协议下 Sapiens2-5B 为 57.83%。COCO 手部 AP 为 64.81，比基座高 12.89 个百分点。宏平均取决于论文的手部裁剪与归一化规程。作者在 HF 发布增量权重，加载器校验并叠加到指定的 Sapiens2-5B 基座，Model Zoo 的权重按 Sapiens2 许可发布。项目当前单仓模型文件清单不具备这套基座与增量的依赖表达。[论文表 4–5](https://arxiv.org/html/2611.10416) · [发布与基座说明](https://github.com/ronggong000/S2HAND#checkpoints) · [固定增量权重](https://huggingface.co/mage0001/S2HAND/tree/7a6c582ecff34ef5fa41a8c941b89ce0780b8511) · [增量权重协议](https://huggingface.co/mage0001/S2HAND/blob/7a6c582ecff34ef5fa41a8c941b89ce0780b8511/LICENSE.md)
- **DWPose-l 384 × 288** 同一官方 COCO-WholeBody 评测的 **Hand AP 为 62.1**，可作为二维手点 Apache-2.0 候选。OpenMMLab 的 RTMW-x 384 × 288 使用其官方 COCO-WholeBody 配置报告 **Hand AP 66.4、Whole AP 70.2**，一手权重位于 OpenMMLab 下载域，可通过其模型配置追溯。以上 AP 与 S2HAND 的跨数据集 AUC 使用不同口径。[DWPose 官方结果](https://github.com/IDEA-Research/DWPose#results) · [DWPose 固定权重](https://huggingface.co/yzd-v/DWPose/tree/1a7144101628d69ee7a3768d1ee3a094070dc388) · [RTMW 官方指标及权重清单](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.yml)

## 语音合成

- **Fish Audio Drama 3**。Artificial Analysis 的 Provider Voice 开放权重榜以模型自带音色做人类成对偏好评估，本次观察为 **第 1，Elo 1279 ± 16**，1,990 项样本、8 种音色。官方权重是 35B MoE（激活 3B），采用 Fish Audio Community License v1.0。商业条款给符合条件的年营收低于 50 万美元主体有限授权，许可定义的 Hosted Service 需单独书面授权。HF 精确许可和版本以仓库协议文本为准，项目当前许可名单未收录该协议。[AA 原生音色榜](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights) · [技术报告](https://arxiv.org/abs/2610.18010) · [固定权重](https://huggingface.co/fishaudio/drama-3/tree/78e5ab79e49aa9a25abe8799eff2c4d73a1c3216) · [权重协议](https://huggingface.co/fishaudio/drama-3/blob/78e5ab79e49aa9a25abe8799eff2c4d73a1c3216/LICENSE)
- **Ming-3.0-tts-tiny**。官方卡片以 WER 标注 Seed-TTS test-zh **0.820%**、test-en **1.666%**，对应 SIM 为 **0.781、0.728**；在另一个五语种零样本表内，它的中英文 WER **0.325%、0.971%** 为所列模型最低。独立 AA 榜的 Ming-3.0-tts 系列名录为第 4、1182 ± 12，未绑定到 tiny 权重。tiny 为 7.9B 总参数、1.3B 激活的 MIT 权重，含音频 tokenizer 及说话人编码器，由另一个官方 GitHub 仓库提供推理程序。[官方评测和限制](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny) · [固定权重](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny/tree/4715ef5a2f02901f75687d538b6920fd0dd11e6b) · [AA 原生音色榜](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights)

## 视频生成

- **MiniMax H3**。Arena 文生视频 2026-11-08 开放权重筛选 **第 1，1456 ± 8**，29,257 票。AA-Video-T2V v2.1 含音频开放权重榜的 MiniMax H3（768p）**1159 ± 13**，8,225 项样本，点估计第 1，排名区间 1–2。平台测的是同名托管模型。官方有同步生成音视频的 Hugging Face 权重；MiniMax H3 Community License 排除欧盟、英国、美国、韩国，项目当前许可名单未收录。[Arena 文生视频榜](https://arena.ai/leaderboard/video/text-to-video?license=open-source) · [AA 含音频文生视频榜](https://artificialanalysis.ai/video/leaderboard/text-to-video/open-weights) · [固定版本权重](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317) · [权重协议](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/LICENSE)
- **Kandinsky 5.0 Video Pro** 在 Arena 同一快照为 **第 2，1175 ± 21**，2,020 票，榜内采用 MIT 或 Apache-2.0 的最高项。上游提供 MIT 的 5 秒 T2V Pro SFT Diffusers 检查点，Arena 的团队条目未公布究竟使用 5 秒还是其他 Pro 版本。技术报告 §8.2.6–7 还分别比较了 MovieGen 的视觉质量、动态和提示跟随。[Arena 文生视频榜](https://arena.ai/leaderboard/video/text-to-video?license=open-source) · [固定 5 秒 SFT 权重](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/tree/7271a43778448e4756a0a822ceb9e46cf52a42b2) · [Kandinsky 技术报告](https://arxiv.org/html/2511.14993)

## Agent 后续复核来源

- 文生图、单图编辑和文生视频可查 Arena 同一任务的开放权重视图，记录榜单日期、版本、投票数、区间与 Preliminary 状态。AA 的编辑、含音频视频和原生音色榜补充独立平台结果。同一模型的不同榜单 Elo 按各自量表保存。[Arena 排行与方法说明入口](https://arena.ai/leaderboard) · [AA 方法说明入口](https://artificialanalysis.ai/methodology)
- 翻译可持续查看 WMT General MT 主办方的赛道、数据和 Findings，逐语言方向记录人评统计组，并区分提交系统与单个检查点。二维人体、手部和换脸优先查对应论文表格与作者发布，记录关键点定义、裁剪/检测条件、测试集和指标方向。[WMT26 官方赛道](https://www2.statmt.org/wmt26/translation-task.html) · [WMT26 Findings](https://aclanthology.org/2026.wmt-main.48/)
- 权重由发布者页面交叉确认仓库归属。HF `GET /api/models/{repo_id}?blobs=true` 可匿名获得 `sha`、`gated`、`cardData`、`siblings` 和文件大小。把榜单证据版本和下载提交分别记录。自定义协议可同时出现在 `license=other`、`license_name` 与单独许可证文件里。该接口已抽查实际返回。当前评测平台来源使用公开网页，未验证可长期调用的结果 API。[HF Hub API 文档](https://huggingface.co/docs/hub/api) · [权重元数据示例](https://huggingface.co/api/models/Qwen/Qwen-Image-2.1?blobs=true)
