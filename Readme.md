# Temporal Consistent Automatic Video Colorization via Semantic Correspondence ![visitors](https://visitor-badge.glitch.me/badge?page_id=bupt-ai-cz.TCVC)
[Project](https://bupt-ai-cz.github.io/TCVC/)
----------
This is the code for paper "Temporal Consistent Automatic Video Colorization via Semantic Correspondence"

Our method achieves the 3rd place in NTIRE 2023 Video Colorization Challenge, Track 2: Color Distribution Consistency (CDC) Optimization

To run the test code, please modify the "--data_root_val" in ./stage1/test.py  and  "--test_path" in ./stage2/inference_colorvid.py to the path of the test dataset.

Please run test.sh to test the model.

    bash test.sh

The checkpoint 342000 in stage1 is finetuned for the NTIRE2023 video colorization challenge. And 340000 is the model we used in the paper.

Any problem about the implementation, please contact sqchen@bupt.edu.cn



## related articles

    @misc{exemplarvcld,      
        title={Exemplar-based Video Colorization with Long-term Spatiotemporal Dependency},       
        author={Siqi Chen and Xueming Li and Xianlin Zhang and Mingdao Wang and Yu Zhang and Jiatong Han and Yue Zhang},      
        year={2023},      
        eprint={2303.15081},      
        archivePrefix={arXiv}}

