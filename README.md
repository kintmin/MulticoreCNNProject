# MulticoreCNNProject
  세종대학교 멀티코어프로그래밍 프로젝트
  
  ![image](https://user-images.githubusercontent.com/40791869/215701734-6c011025-16af-4b8e-8cf5-ded128370812.png)
  OpenCL 병렬 프로그래밍을 활용한 CNN(Convolutional Neural Network) 가속화하여 원하는 결과를 빠르게 도출해내는 프로젝트입니다.

  <br/>
  
## 목차
  1. 기술 스택
  2. 구현 기능
  3. 실행 결과
  4. 배운 점
  
  <br/>
  
## 기술 스택
  - OpenCL
  - 프로그래밍 언어 : C

  <br/>
  
## 구현 기능
  - Convolution Layer
    - 이미지를 분류하는데 필요한 정보를 필터를 통해 뽑아냄
    - 6중 반복문을 2중 반복문으로 병렬화
    - global size : {d1, d2 * n * n * batch_num}
    - local size : {d1, 1}
  
  - Pooling Layer
    - Convolution Filter를 거친 결과에서 해당 영역 내에서 가장 큰 값을 도출
    - 5중 반복문을 2중 반복문으로 병렬화
    - global size = {batch_num, d * n * n}
  
  - FC Layer
    - Pooling Layer의 값으로 이미지 분류
    - 2중 반복문을 모두 병렬화
    - global size = {input * weight}
    - local size = N
  
  - Reduction
    - 각 barrier마다 1개씩 더하는 방법에서 3개씩 더하여 barrier waiting 줄임
  
  - image batch
    - 작업 수행 메모리 크기가 적어 동시에 여러 개의 이미지를 처리
  
  - buffer 메모리 최적화
    각 함수마다 input 메모리가 output 되는 것을 발견하고 메모리 버퍼를 스왑하여 readbuffer 오버헤드를 줄임
    
  <br/>
  
## 실행 결과
  RTX 3060으로 측정하여 600초에서 15.01초로 줄였습니다.
  
  <br/>
  
## 배운 점
  - OpenCL의 디버깅이 어려워서 IDE를 믿는 것이 아니라 어디가 잘못 됐는지 직접 생각하여 디버깅을 함
    - 코드의 알고리즘을 검토할 수 있었고, 어느 부분에서 잘못 됐는지 파악할 수 있는 능력을 키웠습니다.
    
  - 최근 멀티코어에 대한 논문 10개를 읽으면서 전공의 최근 동향을 파악하는데 논문이 큰 도움이 된다는 것을 알았습니다.
