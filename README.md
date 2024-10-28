이 저장소에 있는 mobile_net.py 파일은<br/>
Mobilenet v1, v2, v3을 모두 구현이 가능하게끔 작성한 파일이며,<br/>
<br/>
mobilenet v1 : https://arxiv.org/abs/1704.04861<br/>
mobilenet v2 : https://arxiv.org/abs/1801.04381<br/>
mobilenet v3 : https://arxiv.org/abs/1905.02244<br/>
<br/>
위 3개의 모델을 소개한 논문에서 제시하고 있는<br/>
각 버전별 파생모델을 구현할 수 있습니다.<br/>
<br/>
모델 구현 방식은 해당 py파일의 __main__ 구문을 참조 바라며<br/>
공통적으로 모델의 인자 값 중 'Feature_ext_set' 인자에<br/>
각 버전별 모델 세팅값(V1_setting, V2_setting, V3_setting)을 입력하면<br/>
각 버전별로 모델을 객체화 할 수 있습니다<br/>
<br/>
이때 V3_setting은 ['Large']버전과 ['Small'] 두가지 버전이 존재하니 이를 유의 바랍니다.<br/>
<br/>
모든 모델의 버전은 width_multiplier 옵션을 가변적으로 지정이 가능합니다.<br/>
