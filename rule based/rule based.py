# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:12:32 2022

@author: user
"""

from durable.lang import *

with ruleset('Autonomous Driving'):
    # 규칙 생성
    @when_all((m.explain == "원격 주행하기 위해") & (m.do == "여러 개의 장비를 설치한다."))
    def ruel1(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '원격 주행하기 위해', 'do': '여러 개의 장비를 설치한다.'})
        
    @when_all((m.explain == "장비는") & (m.do == "Camera, Lidar, Ladar 으로 나눈다."))
    def ruel2(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '장비는', 'do': 'Camera, Lidar, Ladar 으로 나눈다.'})
                       
    @when_all((m.explain == "시야 확보를 위해") & (m.do == "4개로 이루어 진다."))
    def ruel3(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '시야 확보를 위해', 'do': '4개로 이루어 진다.'})

    @when_all((m.explain == "레이저를 통해") & (m.do == "물체를 감지한다."))
    def ruel4(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '레이저를 통해', 'do': '물체를 감지한다.'})
        
    @when_all((m.explain == "초음파를 통해") & (m.do == "물체를 감지한다."))
    def ruel5(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '초음파를 통해', 'do': '물체를 감지한다.'})

    @when_all((m.explain == "Camera의 출력을 통해") & (m.do == "주행한다."))
    def ruel6(c):
        c.assert_fact({'subject': c.mSubject, 'explain': 'Camera의 출력을 통해', 'do': '주행한다.'})
        
    @when_all((m.explain == "핵심은") & (m.do == "인터넷 통신이다."))
    def ruel7(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '핵심은', 'do': '인터넷 통신이다.'})
        
    @when_all((m.explain == "IMG 파일의 크기는") & (m.do == "640x640 사이로 조정한다."))
    def ruel8(c):
        c.assert_fact({'subject': c.mSubject, 'explain': 'IMG 파일의 크기는', 'do': '640x640 사이로 조정한다.'})
        
    @when_all((m.explain == "IMG 파일의 확장자는 ") & (m.do == "PNG로 변경한다."))
    def ruel9(c):
        c.assert_fact({'subject': c.mSubject, 'explain': 'IMG 파일의 확장자는', 'do': 'PNG로 변경한다.'})
        
    @when_all((m.explain == "GPU는 ") & (m.do == "최대 4개를 이용한다."))
    def ruel10(c):
        c.assert_fact({'subject': c.mSubject, 'explain': 'GPU는', 'do': '최대 4개를 이용한다.'})
        
    @when_all((m.explain == "학습 매개변수를 ") & (m.do == "조정한다."))
    def ruel11(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '학습 매개변수를', 'do': '조정한다.'})
        
    @when_all((m.explain == "IMG 파일의 크기는 ") & (m.do == "416x416 사이즈로 조정한다."))
    def ruel12(c):
        c.assert_fact({'subject': c.mSubject, 'explain': 'IMG 파일의 크기는 ', 'do': '416x416 사이즈로 조정한다.'})
        
    @when_all((m.explain == "비교적 가벼운 ") & (m.do == "tiny 모델을 이용한다."))
    def ruel13(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '비교적 가벼운 ', 'do': 'tiny 모델을 이용한다.'})
        
    @when_all((m.explain == "원거리 ") & (m.do == "물체를 감지한다."))
    def ruel14(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '원거리 ', 'do': '물체를 감지한다.'})
        
    @when_all((m.explain == "근거리 ") & (m.do == "물체를 감지한다."))
    def ruel15(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '근거리 ', 'do': '물체를 감지한다.'})
        
    @when_all((m.explain == "Testset을 이용하여 ") & (m.do == "성능을 평가한다."))
    def ruel16(c):
        c.assert_fact({'subject': c.mSubject, 'explain': 'Testset을 이용하여 ', 'do': '성능을 평가한다.'})
        
    @when_all((m.explain == "도로 주행을 통해 ") & (m.do == "성능을 평가한다."))
    def ruel17(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '도로 주행을 통해 ', 'do': '안전성을 평가한다.'})
        
    @when_all((m.explain == "공인 인증을 통해 ") & (m.do == "성능을 평가한다."))
    def ruel18(c):
        c.assert_fact({'subject': c.mSubject, 'explain': '공인 인증을 통해 ', 'do': '안전성을 평가한다.'})
        
assert_fact('Autonomous Driving', {'subject': '자율주행차는', 'explain': '원격 주행하기 위해', 'do':'여러 개의 장비를 설치한다.'})
assert_fact('Autonomous Driving', {'subject': '자율주행차의', 'explain': '장비는', 'do': 'Camera, Lidar, Ladar 으로 나눈다.'})
assert_fact('Autonomous Driving', {'subject': 'Camera는', 'explain': '시야 확보를 위해', 'do': '4개로 이루어 진다.'})
assert_fact('Autonomous Driving', {'subject': 'Lidar는', 'explain': '레이저를 통해', 'do': '물체를 감지한다.'})
assert_fact('Autonomous Driving', {'subject': 'Ladar는', 'explain': '초음파를 통해', 'do': '물체를 감지한다.'})
assert_fact('Autonomous Driving', {'subject': '원격제어는', 'explain': 'Camera의 출력을 통해', 'do': '주행한다.'})
assert_fact('Autonomous Driving', {'subject': '원격제어의', 'explain': '핵심은', 'do': '인터넷 통신이다.'})
assert_fact('Autonomous Driving', {'subject': '학습 시', 'explain': 'IMG 파일의 크기는', 'do': '640x640 사이로 조정한다.'})
assert_fact('Autonomous Driving', {'subject': '학습 시', 'explain': 'IMG 파일의 확장자는', 'do': 'PNG로 변경한다.'})
assert_fact('Autonomous Driving', {'subject': '학습 시', 'explain': 'GPU는', 'do': '최대 4개를 이용한다.'})
assert_fact('Autonomous Driving', {'subject': '프레임 수 증가를 위해', 'explain': '학습 매개변수를', 'do': '조정한다.'})
assert_fact('Autonomous Driving', {'subject': '프레임 수 증가를 위해', 'explain': 'IMG 파일의 크기는 ', 'do': '416x416 사이즈로 조정한다.'})
assert_fact('Autonomous Driving', {'subject': '프레임 수 증가를 위해', 'explain': '비교적 가벼운 ', 'do': 'tiny 모델을 이용한다.'})
assert_fact('Autonomous Driving', {'subject': 'Lidar의 학습 시', 'explain': '원거리 ', 'do': '물체를 감지한다.'})
assert_fact('Autonomous Driving', {'subject': 'Ladar의 학습 시', 'explain': '근거리 ', 'do': '물체를 감지한다.'})
assert_fact('Autonomous Driving', {'subject': '학습 후', 'explain': 'Testset을 이용하여 ', 'do': '성능을 평가한다.'})
assert_fact('Autonomous Driving', {'subject': '학습 후', 'explain': '도로 주행을 통해 ', 'do': '안전성을 평가한다.'})
assert_fact('Autonomous Driving', {'subject': '자율주행차는', 'explain': '공인 인증을 통해 ', 'do': '안전성을 평가한다.'})