# coding=utf-8


def getFilteredText(prd_nm, keyword_info, prd_info, mall_nm):

    text = prd_nm.strip() + ' '  + getFilteredKeywordInfo(keyword_info, prd_nm) + ' ' + getFilteredPrdInfo(prd_info, prd_nm)

    text = text.replace("\n", " ").replace("!", "").strip()

    return text


def getFilteredKeywordInfo(keyword_info, prd_nm):

    if(keyword_info is None):
        return ''
    else:
        return keyword_info.strip().replace(",", " ")

def getFilteredPrdInfo(prd_info, prd_nm) :
    if (prd_info is None):
        return ''

    isFiltred = False
    if(prd_info=='상품 옵션'):
        isFiltred = True
    elif(prd_info=='여성의류, 원피스, 블라우스, 니트, 스커트, 야상 등 판매 쇼핑몰'):
        isFiltred = True
    elif(prd_info=='10대, 20대,에이인 여성의류 쇼핑몰, 유니크 스타일, 유니섹스, 트랜디 편집샵 원피스, AIN.'):
        isFiltred = True
    elif(prd_info=='10대,20대 쇼핑몰,유니크,감성적,심플,모던,베이직,데일리,데일리룩,스커트등 판매'):
        isFiltred = True
    elif(prd_info=='여성의류, 자체제작 전문 쇼핑몰, 로맨틱 데이트룩, 데일리룩'):
        isFiltred = True
    elif(prd_info=='합리적인 가격의 데일리룩 20대 여자 쇼핑몰,데일리어바웃,스키니나인,신상5%할인'):
        isFiltred = True
    elif(prd_info=='10대-20대 남여공용쇼핑몰, 유니크,데일리,캐주얼, 러블리 스타일, 자체제작 쇼핑몰'):
        isFiltred = True
    elif(prd_info=='모델 임지혜의 일상정보'):
        isFiltred = True
    elif (prd_info == '여성의류 전문 쇼핑몰, 티셔츠, 블라우스, 카디건, 팬츠, 원피스 판매'):
        isFiltred = True
    elif (prd_info == '매일매일 핫한 변신을 꿈꾸는 너에게 its you, 머리부터 발끝까지 츄 chuu"'):
        isFiltred = True
    elif (prd_info == '매력적인 명품 & 섹시 클럽의상, 홀복 전문 쇼핑몰'):
        isFiltred = True
    elif (prd_info == '10, 20대 여자쇼핑몰, 유니크스타일, 데일리룩, 와이드슬랙스, 멜빵바지.'):
        isFiltred = True
    elif (prd_info == '유니크 데일리룩 st. 반하루 20대 여자 의류 쇼핑몰, 데일리룩 코디, 유니크 패션, 프렌치 스트릿 스타일.'):
        isFiltred = True
    elif (prd_info == '여성의류 전문 쇼핑몰, 아우터, 티셔츠, 원피스, 바지, 치마, 신발, 가방, 액세서리 등 판매.'):
        isFiltred = True
    elif (prd_info == '여성의류 쇼핑몰, 수입보세, 명품스타일, 니트, 자켓 등 판매.'):
        isFiltred = True
    elif (prd_info == '여성의류쇼핑몰,여자옷,원피스,코트,블라우스,레깅스,드레스,여성옷'):
        isFiltred = True
    elif (prd_info == '착한가격 칭찬해! 매일 찾게되는 감각있는 데일리룩'):
        isFiltred = True
    elif (prd_info == '여성의류 쇼핑몰, 예쁜 자체제작 자켓, 오프숄더, 블라우스, 슬랙스 팬츠, 스커트, 원피스, 셀프웨딩, 판매'):
        isFiltred = True
    elif (prd_info == '쏭바이쏭 쏭언니의 감성이 듬뿍담긴 자체제작NO.1 감성라벨 여성의류 쇼핑몰.'):
        isFiltred = True
    elif (prd_info == '가장 아름다운 여성속옷,란제리,비키니,래시가드 등 언더웨어판매 1위 쇼핑몰'):
        isFiltred = True
    #     179

    if(isFiltred == True):
        # print("isFiltred = True")
        # prd_info = prd_nm
        return ''

    return prd_info.strip().replace(",", " ")

