# 인자값 순서(Kernel, t, c, SE, AF, S)
# k: 커널사이즈(3x3 or 5x5)
# t: 확장 계수 (expand ratio)
# c: 출력 채널 수 
# SE : SE블럭 삽입 유/무
# AF : 적용할 활성화 함수 종류
# s: 스트라이드 (S = 1 or 2)
# BN : Batchnormal 레이어 유/무
V3_setting = {
    'Large' : [
        [
            [3, 1,    16, False, 'RL', 1],
            [3, 4,    24, False, 'RL', 2],
            [3, 3,    24, False, 'RL', 1],
            [5, 3,    40,  True, 'RL', 2],
            [5, 3,    40,  True, 'RL', 1],
            [5, 3,    40,  True, 'RL', 1],
            [3, 6,    80, False, 'HS', 2],
            [3, 2.5,  80, False, 'HS', 1],
            [3, 2.3,  80, False, 'HS', 1],
            [3, 2.3,  80, False, 'HS', 1],
            [3, 6,   112,  True, 'HS', 1],
            [3, 6,   112,  True, 'HS', 1],
            [5, 6,   160,  True, 'HS', 1],
            [5, 6,   160,  True, 'HS', 1],
        ],  #k, t,    c,    SE,   AF,  s
        # Efficient Last Stage 인자값
        [   #i_c, o_ch    BN,   SE,    AF
            [160, 960,  True,  False, 'HS'],
            [960, 1280, False, False, 'HS'],
            [1280, -1,  False, False, None],
        ]
    ],  

    'Small' : [
        [
            [3, 1,    16,  True, 'RL', 2],
            [3, 4.5,  24, False, 'RL', 2],
            [3, 11/3, 24, False, 'RL', 1],
            [5, 4,    40,  True, 'HS', 2],
            [5, 6,    40,  True, 'HS', 1],
            [5, 6,    40,  True, 'HS', 1],
            [5, 3,    48,  True, 'HS', 1],
            [5, 3,    48,  True, 'HS', 1],
            [5, 6,    96,  True, 'HS', 2],
            [5, 6,    96,  True, 'HS', 1],
            [5, 6,    96,  True, 'HS', 1],
        ],  #k, t,    c,    SE,   AF,  s
        # Efficient Last Stage 인자값
        [   #i_c, o_ch    BN,   SE,    AF
            [96,  576,  True,   True, 'HS'],
            [576, 1024, False, False, 'HS'],
            [1024, -1,  False, False, None],
        ]
    ]
}

# MobileNet v2설정 계수값 (t, c, n, s)
# t: 확장 계수 (expand ratio)
# c: 출력 채널 수 
# n: 반복 횟수
# AF: 활성화 함수는 ReLU6
# s: 스트라이드 (S = 1 or 2)
V2_setting = [
    [
        #t,   c, n,  AF,  s
        [1,  16, 1, 'R6', 1],
        [6,  24, 2, 'R6', 2],
        [6,  32, 3, 'R6', 2],
        [6,  64, 4, 'R6', 2],
        [6,  96, 3, 'R6', 1],
        [6, 160, 3, 'R6', 2],
        [6, 320, 1, 'R6', 1],
    ],
    # Last Stage 인자값
    [   #i_c,  o_ch   BN,    SE,   AF
        [320,  1280, True, False, 'R6'],
        [1280,  -1,  True, False, 'R6'],
    ]
    
]

# MobileNet v1설정 계수값 (c, n, AF, s)
# c: 출력 채널 수 
# n: 반복 횟수
# AF : 모두 ReLU6
# s: 스트라이드 (S = 1 or 2)
V1_setting = [
    #  c,  n,  AF,  s
    [  64, 1, 'R6', 1],
    [ 128, 2, 'R6', 2],
    [ 256, 2, 'R6', 2],
    [ 512, 6, 'R6', 2],
    [1024, 2, 'R6', 2],
]


import torch
import torch.nn as nn

from thop import profile, clever_format

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, AF=None, **kwargs):
        super(BasicConv, self).__init__()

        # padding='same'인 경우 padding 값 자동 계산
        if padding == 'same':
            padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                      padding=padding, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch)
        ]
        # 활성화 함수는 총 3가지 조건으로 적용
        # Default=None인 경우 활셩화 함수 적용 X
        if AF == 'RL':
            layers.append(nn.ReLU(inplace=True))
        elif AF == 'R6':
            layers.append(nn.ReLU6(inplace=True))
        elif AF == 'HS':
            layers.append(nn.Hardswish(inplace=True))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x
    
class DepthSep(nn.Module):
    def __init__(self, in_ch, out_ch, stride, AF=None):
        super(DepthSep, self).__init__()

        # depth-wise Conv는 groups 옵션으로 구현한다.
        self.depthwise = BasicConv(in_ch, in_ch, kernel_size=3, 
                                   stride=stride, padding='same',
                                   AF = AF, groups = in_ch)
         # point-wise Conv는 [1x1]커널로 구현한다.
        self.pointwise = BasicConv(in_ch, out_ch, kernel_size=1, 
                                   stride=1, padding=0, AF = AF)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x
    
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super(SEBlock, self).__init__()

        # Squeeze 단계: 채널의 글로벌 정보 획득 
        self.squeeze_path = nn.AdaptiveAvgPool2d((1, 1))

        # Excitation 단계: 채널 중요도 계산
        # (H, W)가 모두 (1,1)되었기에 [1x1]nn.conv2D는 nn.Linear와
        # 같은 연산 결과를 기대할 수 있게 됨
        # 여기서 [1x1]conv2d는 flatten을 안써도 되고 병렬연산에 더 친화적
        # 따라서 FC layer은 동일연산의 [1x1]conv2d로 변경
        self.excitation_path = nn.Sequential(
            nn.Conv2d(ch, ch//reduction, 1),    # 채널 크기 감소
            nn.ReLU(inplace = True),            # 비선형성 추가
            nn.Conv2d(ch//reduction, ch, 1),    # 채널 크기 원상 복원
            nn.Hardsigmoid(inplace = True)      # 중요도 계산
        )

    def forward(self, x):
        # 1. Squeeze 단계: (BS, C, H, W) -> (BS, C, 1, 1)
        y = self.squeeze_path(x)
        # 2. Excitation 단계: Attention Score를 적용한 (BS, C, 1, 1)
        y = self.excitation_path(y)
        # 3. Recalibration 단계: 채널별 중요도(Attention Score)
        # 를 입력 텐서에 곱함, 여기서 expands_as로 (bs, ch, H, W) 확장
        Recalibration = x * y.expand_as(x)

        return Recalibration
    

class BNeckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, 
                 AF=None, SE=False):
        super(BNeckBlock, self).__init__()

        # 중간 블럭의 채널개수(차원)은 확장계수로 조정됨
        hidden_dim = int(in_ch * expand_ratio)

        # Residual Connection이 활성화되는 조건
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = [] #쌓을 레이어를 리스트에 담음

        if expand_ratio != 1:
            # 확장 단계 : 커널 사이즈가 1 + 채널크기가 변경되는 Point-wies Conv
            layers.append(BasicConv(in_ch, hidden_dim, kernel_size=1,
                                    stride=1, padding=0, AF = AF))
        
        # Depthwise Conv -> groups 옵션을 사용하여 커널을 그룹별로 적용
        layers.append(BasicConv(hidden_dim, hidden_dim,
                                kernel_size=kernel_size, 
                                stride=stride, padding='same',
                                AF = AF, groups=hidden_dim))
        
        if SE: # SE_block옵션이 True이면 SEblock를 추가한다
            layers.append(SEBlock(hidden_dim))
        
        # 축소단계 : 확장한 채널을 다시 감소시킴, 이때 활성화함수 적용X
        # 이 부분 선형 Point-wise Conv이다.
        layers.append(BasicConv(hidden_dim, out_ch, kernel_size=1, 
                                stride=1, padding=0, AF = None))
        
        self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.res_block(x)
        else:
            return self.res_block(x)
        
class LastStage(nn.Module):
    def __init__(self, alpha=1.0, classifier_set = None,
                 num_classes=1000):
        super(LastStage, self).__init__()
        layers = []

        for in_ch, out_ch, BN, SE, AF in classifier_set:
            in_ch = int(in_ch * alpha)
            if out_ch > 0:
                out_ch = int(out_ch * alpha)
            else:
                out_ch = num_classes
            
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                    stride=1, padding=0, bias=False))
            if BN: #Batch_norm 이 있으면 해당 레이어 추가
                layers.append(nn.BatchNorm2d(out_ch))
            if SE: #SE_block이 있으면 해당 블럭 추가
                layers.append(SEBlock(out_ch))
            
            # 활성화 함수는 조건별로 삽입
            if AF == 'RL':
                layers.append(nn.ReLU(inplace=True))
            elif AF == 'R6':
                layers.append(nn.ReLU6(inplace=True))
            elif AF == 'HS':
                layers.append(nn.Hardswish(inplace=True))

        # 완성된 Layers 리스트의 1번째 자리에 AvgPool 넣기
        layers.insert(1, nn.AdaptiveAvgPool2d((1,1)))
        # 가장 마지막 레이어에 Flatten 레이어 넣기
        layers.append(nn.Flatten())

        self.Last_stage_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.Last_stage_block(x)

        return x
    

class MobileNetV1(nn.Module):
    def __init__(self, width_multiplier=1.0, Feature_ext_set = None,
                 num_classes=1000):
        super(MobileNetV1, self).__init__()
        #네트워크 각 층의 필터 개수를 조정하는 인자값
        self.alpha = width_multiplier

        # stem의 out_ch이자 Feature_ext의 Start in_ch
        in_ch = int(32*self.alpha)
        self.stem = BasicConv(3, in_ch, kernel_size=3, 
                              stride=2, padding=1, AF='R6')
        
        blocks = []

        for c, n, AF, s in Feature_ext_set:
            out_ch = int(c * self.alpha)
            for i in range(n):
                # 첫 번째 블록만 stride=s에 따라 설정, 나머지는 stride=1
                stride = s if i == 0 else 1
                blocks.append(DepthSep(in_ch, out_ch, 
                                       stride=stride, AF=AF))
                in_ch = out_ch  # 블럭 생성 후 in_ch를 업데이트

        self.feature_ext = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_ext(x)
        x = self.classifier(x)

        return x
    

class MobileNetV2(nn.Module):
    def __init__(self, width_multiplier=1.0, Feature_ext_set = None,
                 num_classes=1000):
        super(MobileNetV2, self).__init__()
        #네트워크 각 층의 필터 개수를 조정하는 인자값
        self.alpha = width_multiplier

        # stem의 out_ch이자 Feature_ext의 Start in_ch
        in_ch = int(32*self.alpha)
        self.stem = BasicConv(3, in_ch, kernel_size=3, 
                              stride=2, padding=1, AF='R6')
        
        blocks = []
        # 레이어 설계 세팅값이 정상적으로 들어올 경우 기동
        if len(Feature_ext_set[0][0]) == 5:
            for t, c, n, AF, s in Feature_ext_set[0]:
                out_ch = int(c * self.alpha)
                for i in range(n):
                    # 첫 번째 블록만 stride=s에 따라 설정, 나머지는 stride=1
                    stride = s if i == 0 else 1
                    blocks.append(BNeckBlock(in_ch=in_ch, out_ch=out_ch,
                                             kernel_size=3, stride=stride,
                                             expand_ratio=t, AF=AF))
                    in_ch = out_ch  # 블럭 생성 후 in_ch를 업데이트
        self.feature_ext = nn.Sequential(*blocks)

        self.classifier = LastStage(self.alpha, Feature_ext_set[1], 
                                    num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_ext(x)
        x = self.classifier(x)

        return x
    

class MobileNetV3(nn.Module):
    def __init__(self, width_multiplier=1.0, Feature_ext_set = None,
                 num_classes=1000):
        super(MobileNetV3, self).__init__()
        #네트워크 각 층의 필터 개수를 조정하는 인자값
        self.alpha = width_multiplier

        # stem의 out_ch이자 Feature_ext의 Start in_ch
        in_ch = int(16*self.alpha)
        self.stem = BasicConv(3, in_ch, kernel_size=3, 
                              stride=2, padding=1, AF='HS')
        
        blocks = []
        # 레이어 설계 세팅값이 정상적으로 들어올 경우 기동
        if len(Feature_ext_set[0][0]) == 6:
            for k, t, c, SE, AF, s in Feature_ext_set[0]:
                out_ch = int(c * self.alpha)
                blocks.append(BNeckBlock(in_ch=in_ch, out_ch=out_ch,
                                         kernel_size=k, stride=s,
                                         expand_ratio=t, AF = AF, SE = SE))
                in_ch = out_ch # 세팅된 블럭 추가 후 in_ch업데이트
        self.feature_ext = nn.Sequential(*blocks)

        self.classifier = LastStage(self.alpha, Feature_ext_set[1], 
                                    num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_ext(x)
        x = self.classifier(x)

        return x
    

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## 설계한 모델의 디버그 함수
def mobile_debug(model, resolution_multiplier):

    img_size = 0

    # resolution_multiplier의 타입을 체크하여 이미지 크기를 계산
    # 0~1사이의 비율로 입력했을 시
    if isinstance(resolution_multiplier, float):
        img_size = int(224 * resolution_multiplier)
    # 입력 이미지 사이즈로 입력했을 시
    elif isinstance(resolution_multiplier, int):
        img_size = resolution_multiplier

    
    # 모델을 평가 모드로 전환
    model.eval()

    if next(model.parameters()).is_cuda:
        # 더미 입력 데이터 생성
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    else:
        dummy_input = torch.randn(1, 3, img_size, img_size)

    # FLOPs 및 파라미터 수 계산
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")

    # 결과 출력
    print(f"모델의 FLOPs: {flops}, Params: {params}")


if __name__ == '__main__':
    # MobileNetV3 인스턴스화 및 디버그
    print("=== MobileNetV3 ===")
    model_v3_L = MobileNetV3(width_multiplier=1.0, 
                            Feature_ext_set=V3_setting['Large'], 
                            num_classes=1000).to(device)
    mobile_debug(model_v3_L, resolution_multiplier=0.5)

    model_v3_S = MobileNetV3(width_multiplier=1.0, 
                            Feature_ext_set=V3_setting['Small'], 
                            num_classes=1000).to(device)
    mobile_debug(model_v3_S, resolution_multiplier=192)

    # MobileNetV2 인스턴스화 및 디버그
    print("\n=== MobileNetV2 ===")
    model_2 = MobileNetV2(width_multiplier=1.0, 
                        Feature_ext_set=V2_setting, 
                        num_classes=1000).to(device)
    mobile_debug(model_2, resolution_multiplier=224)

    # MobileNetV1 인스턴스화 및 디버그
    print("\n=== MobileNetV2 ===")
    model_1 = MobileNetV1(width_multiplier=1.0, 
                        Feature_ext_set=V1_setting, 
                        num_classes=1000).to(device)
    mobile_debug(model_1, resolution_multiplier=1.0)