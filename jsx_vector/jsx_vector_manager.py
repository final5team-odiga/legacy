import os
import json
import ast
import re
from typing import List, Dict, Any
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, VectorSearch,
    VectorSearchProfile, HnswAlgorithmConfiguration, SearchField
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from pathlib import Path

# 환경 변수 로드
dotenv_path = Path(r'C:\Users\EL0021\Desktop\jsx_vector\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class JSXVectorManager:
    """JSX 컴포넌트 벡터 데이터 관리자 - Azure AI Search 활용"""
    
    def __init__(self):
        # Azure 서비스 초기화
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")
        self.jsx_index_name = "jsx-component-vector-index"
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.jsx_index_name,
            credential=AzureKeyCredential(self.search_key)
        )
        
        self.search_index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=AzureKeyCredential(self.search_key)
        )
        
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.embedding_model = "text-embedding-ada-002"
        
    def initialize_jsx_search_index(self):
        """JSX 컴포넌트 특화 검색 인덱스 초기화"""
        try:
            # 기존 인덱스 확인
            try:
                existing_index = self.search_index_client.get_index(self.jsx_index_name)
                print(f"✅ 기존 JSX 인덱스 '{self.jsx_index_name}' 발견")
                
                if self._check_jsx_index_has_data():
                    print(f"✅ JSX 인덱스에 데이터 존재 - 처리 생략")
                    return True
                else:
                    print(f"⚠️ JSX 인덱스는 있지만 데이터 없음 - 컴포넌트 처리 필요")
                    return False
                    
            except Exception as e:
                print(f"📄 기존 JSX 인덱스 없음 - 새로 생성: {e}")
            
            # 새 JSX 인덱스 생성
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="jsx-component-profile",
                        algorithm_configuration_name="jsx-algorithm"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="jsx-algorithm",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            )
            
            fields = [
                # 기본 식별 정보
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="component_name", type=SearchFieldDataType.String, filterable=True, searchable=True),
                SimpleField(name="component_category", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
                
                # JSX 구조 정보
                SimpleField(name="jsx_structure", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="layout_method", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="responsive_strategy", type=SearchFieldDataType.String, filterable=True),
                
                # 이미지 패턴
                SimpleField(name="image_count", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="image_arrangement", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="image_sizing", type=SearchFieldDataType.String, filterable=True),
                
                # 텍스트 패턴
                SimpleField(name="text_hierarchy", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="typography_classes", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="text_alignment", type=SearchFieldDataType.String, filterable=True),
                
                # Tailwind CSS 패턴
                SimpleField(name="color_palette", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="spacing_scale", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="responsive_classes", type=SearchFieldDataType.String, searchable=True),
                
                # 메타데이터
                SimpleField(name="complexity_level", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="reusability_score", type=SearchFieldDataType.Double, filterable=True),
                SimpleField(name="mobile_optimized", type=SearchFieldDataType.Boolean, filterable=True),
                
                # 실제 JSX 코드
                SimpleField(name="jsx_code", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="import_statements", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="export_pattern", type=SearchFieldDataType.String, searchable=True),
                
                # 검색 키워드
                SimpleField(name="search_keywords", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="embedding_text", type=SearchFieldDataType.String, searchable=True),
                
                # 벡터 필드
                SearchField(
                    name="jsx_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="jsx-component-profile"
                )
            ]
            
            index = SearchIndex(
                name=self.jsx_index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.search_index_client.create_index(index)
            print(f"✅ 새 JSX 인덱스 '{self.jsx_index_name}' 생성 완료")
            return False  # 데이터 처리 필요
            
        except Exception as e:
            print(f"❌ JSX 인덱스 초기화 실패: {e}")
            return False
    
    def _check_jsx_index_has_data(self) -> bool:
        """JSX 인덱스에 데이터가 있는지 확인"""
        try:
            results = self.search_client.search(
                search_text="*",
                top=1,
                include_total_count=True
            )
            
            document_count = 0
            for result in results:
                document_count += 1
                break
                
            if document_count > 0:
                print(f"✅ JSX 데이터 확인됨: 최소 {document_count}개 컴포넌트 존재")
                return True
            else:
                print(f"❌ JSX 데이터 없음: 인덱스가 비어있음")
                return False
                
        except Exception as e:
            print(f"❌ JSX 데이터 확인 중 오류: {e}")
            return False
    
    def process_jsx_components(self, components_folder: str = "components"):
        """JSX 컴포넌트 폴더 처리하여 벡터 인덱스 생성"""
        if not os.path.exists(components_folder):
            print(f"❌ 컴포넌트 폴더를 찾을 수 없습니다: {components_folder}")
            return
        
        # 인덱스 초기화 및 데이터 존재 여부 확인
        has_data = self.initialize_jsx_search_index()
        if has_data:
            print("🎉 기존 JSX 벡터 데이터 사용 - 컴포넌트 처리 완전 생략")
            return
        
        print("📄 JSX 컴포넌트 처리 시작")
        
        # JSX 파일 수집
        jsx_files = [f for f in os.listdir(components_folder) if f.endswith('.jsx')]
        
        if not jsx_files:
            print(f"❌ 처리할 JSX 파일을 찾을 수 없습니다: {components_folder}")
            return
        
        print(f"📁 {len(jsx_files)}개 JSX 파일 발견")
        
        all_documents = []
        for jsx_file in jsx_files:
            jsx_path = os.path.join(components_folder, jsx_file)
            print(f"📄 처리 중: {jsx_file}")
            
            try:
                jsx_document = self._analyze_jsx_component(jsx_path, jsx_file)
                if jsx_document:
                    all_documents.append(jsx_document)
                    print(f"✅ {jsx_file}: 벡터 문서 생성 완료")
                else:
                    print(f"⚠️ {jsx_file}: 분석 실패")
                    
            except Exception as e:
                print(f"❌ {jsx_file} 처리 실패: {e}")
        
        # 벡터 인덱스에 업로드
        if all_documents:
            self._upload_jsx_to_search_index(all_documents)
            print(f"✅ 총 {len(all_documents)}개 JSX 컴포넌트가 벡터 인덱스에 저장됨")
            
            # 최종 확인
            final_count = self._get_jsx_document_count()
            print(f"🎯 최종 JSX 인덱스 문서 수: {final_count}개")
        else:
            print("⚠️ 처리된 JSX 문서가 없습니다")
    
    def _analyze_jsx_component(self, jsx_path: str, jsx_file: str) -> Dict:
        """JSX 컴포넌트 파일 분석하여 벡터 문서 생성"""
        try:
            with open(jsx_path, 'r', encoding='utf-8') as file:
                jsx_code = file.read()
            
            # 컴포넌트 이름 추출
            component_name = jsx_file.replace('.jsx', '')
            
            # JSX 코드 분석
            analysis = self._extract_jsx_patterns(jsx_code, component_name)
            
            # 임베딩 텍스트 생성
            embedding_text = self._create_jsx_embedding_text(analysis)
            
            # 벡터 생성
            vector = self._create_jsx_embedding(embedding_text)
            
            # 문서 ID 생성
            doc_id = self._create_safe_jsx_key(component_name)
            
            # JSX 벡터 문서 구성
            jsx_document = {
                "id": doc_id,
                "component_name": component_name,
                "component_category": analysis["category"],
                "source_type": "magazine_template",
                
                # JSX 구조 정보
                "jsx_structure": json.dumps(analysis["structure"]),
                "layout_method": analysis["layout"]["method"],
                "responsive_strategy": analysis["layout"]["responsive"],
                
                # 이미지 패턴
                "image_count": analysis["images"]["count"],
                "image_arrangement": analysis["images"]["arrangement"],
                "image_sizing": analysis["images"]["sizing"],
                
                # 텍스트 패턴
                "text_hierarchy": json.dumps(analysis["text"]["hierarchy"]),
                "typography_classes": json.dumps(analysis["text"]["typography"]),
                "text_alignment": analysis["text"]["alignment"],
                
                # Tailwind CSS 패턴
                "color_palette": json.dumps(analysis["tailwind"]["colors"]),
                "spacing_scale": json.dumps(analysis["tailwind"]["spacing"]),
                "responsive_classes": json.dumps(analysis["tailwind"]["responsive"]),
                
                # 메타데이터
                "complexity_level": analysis["metadata"]["complexity"],
                "reusability_score": analysis["metadata"]["reusability"],
                "mobile_optimized": analysis["metadata"]["mobile_optimized"],
                
                # 실제 JSX 코드
                "jsx_code": jsx_code,
                "import_statements": json.dumps(analysis["code"]["imports"]),
                "export_pattern": analysis["code"]["export"],
                
                # 검색 최적화
                "search_keywords": analysis["keywords"],
                "embedding_text": embedding_text,
                "jsx_vector": vector
            }
            
            return jsx_document
            
        except Exception as e:
            print(f"❌ JSX 분석 실패 ({jsx_file}): {e}")
            return None
    
    def _extract_jsx_patterns(self, jsx_code: str, component_name: str) -> Dict:
        """JSX 코드에서 패턴 추출"""
        
        # 컴포넌트 카테고리 분류
        category = self._classify_component_category(component_name, jsx_code)
        
        # 이미지 패턴 분석
        image_patterns = self._analyze_image_patterns(jsx_code)
        
        # 텍스트 패턴 분석
        text_patterns = self._analyze_text_patterns(jsx_code)
        
        # 레이아웃 패턴 분석
        layout_patterns = self._analyze_layout_patterns(jsx_code)
        
        # Tailwind CSS 패턴 분석
        tailwind_patterns = self._analyze_tailwind_patterns(jsx_code)
        
        # JSX 구조 분석
        structure_patterns = self._analyze_jsx_structure(jsx_code)
        
        # 코드 패턴 분석
        code_patterns = self._analyze_code_patterns(jsx_code)
        
        # 메타데이터 생성
        metadata = self._generate_component_metadata(jsx_code, category)
        
        # 검색 키워드 생성
        keywords = self._generate_search_keywords(category, image_patterns, text_patterns, layout_patterns)
        
        return {
            "category": category,
            "images": image_patterns,
            "text": text_patterns,
            "layout": layout_patterns,
            "tailwind": tailwind_patterns,
            "structure": structure_patterns,
            "code": code_patterns,
            "metadata": metadata,
            "keywords": keywords
        }
    
    def _classify_component_category(self, component_name: str, jsx_code: str) -> str:
        """컴포넌트 카테고리 분류"""
        name_lower = component_name.lower()
        
        if 'image' in name_lower:
            return "image_focused"
        elif 'text' in name_lower:
            return "text_focused"
        elif 'mixed' in name_lower:
            return "mixed"
        elif 'card' in name_lower:
            return "card_based"
        elif 'list' in name_lower:
            return "list_based"
        elif 'dashboard' in name_lower:
            return "dashboard"
        else:
            # JSX 코드 내용으로 판단
            img_count = jsx_code.count('<img')
            text_elements = jsx_code.count('<h1') + jsx_code.count('<h2') + jsx_code.count('<p')
            
            if img_count > text_elements:
                return "image_focused"
            elif text_elements > img_count * 2:
                return "text_focused"
            else:
                return "mixed"
    
    def _analyze_image_patterns(self, jsx_code: str) -> Dict:
        """이미지 패턴 분석"""
        img_count = jsx_code.count('<img')
        
        # 이미지 배치 패턴 분석
        if 'grid' in jsx_code.lower():
            arrangement = "grid"
        elif 'flex' in jsx_code.lower() and img_count > 1:
            arrangement = "flex"
        elif img_count == 1:
            arrangement = "single"
        elif img_count > 3:
            arrangement = "gallery"
        else:
            arrangement = "multiple"
        
        # 이미지 크기 패턴
        if 'width: \'100%\'' in jsx_code:
            sizing = "responsive"
        elif 'aspect-' in jsx_code:
            sizing = "aspect_ratio"
        else:
            sizing = "fixed"
        
        return {
            "count": img_count,
            "arrangement": arrangement,
            "sizing": sizing
        }
    
    def _analyze_text_patterns(self, jsx_code: str) -> Dict:
        """텍스트 패턴 분석"""
        hierarchy = []
        
        if '<h1' in jsx_code:
            hierarchy.append("h1")
        if '<h2' in jsx_code:
            hierarchy.append("h2")
        if '<h3' in jsx_code:
            hierarchy.append("h3")
        if '<p' in jsx_code:
            hierarchy.append("p")
        
        # 타이포그래피 클래스 추출
        typography_matches = re.findall(r'fontSize: [\'"]([^\'"]+)[\'"]', jsx_code)
        typography = list(set(typography_matches))
        
        # 텍스트 정렬 분석
        if 'textAlign: \'center\'' in jsx_code:
            alignment = "center"
        elif 'textAlign: \'right\'' in jsx_code:
            alignment = "right"
        else:
            alignment = "left"
        
        return {
            "hierarchy": hierarchy,
            "typography": typography,
            "alignment": alignment
        }
    
    def _analyze_layout_patterns(self, jsx_code: str) -> Dict:
        """레이아웃 패턴 분석"""
        if 'display: \'grid\'' in jsx_code:
            method = "grid"
        elif 'display: \'flex\'' in jsx_code:
            method = "flexbox"
        elif 'position: \'absolute\'' in jsx_code:
            method = "absolute"
        else:
            method = "block"
        
        # 반응형 전략
        if '@media' in jsx_code or 'responsive' in jsx_code.lower():
            responsive = "responsive"
        else:
            responsive = "fixed"
        
        return {
            "method": method,
            "responsive": responsive
        }
    
    def _analyze_tailwind_patterns(self, jsx_code: str) -> Dict:
        """Tailwind CSS 패턴 분석 (인라인 스타일에서 유추)"""
        colors = []
        spacing = []
        responsive = []
        
        # 색상 패턴 추출
        color_matches = re.findall(r'color: [\'"]([^\'"]+)[\'"]', jsx_code)
        background_matches = re.findall(r'backgroundColor: [\'"]([^\'"]+)[\'"]', jsx_code)
        colors.extend(color_matches + background_matches)
        
        # 간격 패턴 추출
        spacing_matches = re.findall(r'(?:padding|margin): [\'"]([^\'"]+)[\'"]', jsx_code)
        spacing.extend(spacing_matches)
        
        return {
            "colors": list(set(colors)),
            "spacing": list(set(spacing)),
            "responsive": responsive
        }
    
    def _analyze_jsx_structure(self, jsx_code: str) -> Dict:
        """JSX 구조 분석"""
        structure = {
            "component_type": "functional",
            "hooks_used": [],
            "props_pattern": "simple",
            "children_structure": "single"
        }
        
        # Hook 사용 분석
        if 'useState' in jsx_code:
            structure["hooks_used"].append("useState")
        if 'useEffect' in jsx_code:
            structure["hooks_used"].append("useEffect")
        if 'memo' in jsx_code:
            structure["hooks_used"].append("memo")
        
        # 자식 구조 분석
        if jsx_code.count('<div') > 3:
            structure["children_structure"] = "nested"
        elif '?' in jsx_code and ':' in jsx_code:
            structure["children_structure"] = "conditional"
        
        return structure
    
    def _analyze_code_patterns(self, jsx_code: str) -> Dict:
        """코드 패턴 분석"""
        imports = []
        
        # Import 문 추출
        import_matches = re.findall(r'import[^;]+;', jsx_code)
        imports.extend(import_matches)
        
        # Export 패턴
        if 'export default' in jsx_code:
            export_pattern = "default_export"
        else:
            export_pattern = "named_export"
        
        return {
            "imports": imports,
            "export": export_pattern
        }
    
    def _generate_component_metadata(self, jsx_code: str, category: str) -> Dict:
        """컴포넌트 메타데이터 생성"""
        # 복잡도 계산
        complexity_score = (
            jsx_code.count('<div') * 0.5 +
            jsx_code.count('<img') * 1.0 +
            jsx_code.count('style=') * 0.3 +
            jsx_code.count('className=') * 0.2
        )
        
        if complexity_score < 5:
            complexity = "simple"
        elif complexity_score < 15:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        # 재사용성 점수
        reusability = 0.8 if category in ["image_focused", "text_focused"] else 0.6
        
        # 모바일 최적화 여부
        mobile_optimized = 'responsive' in jsx_code.lower() or 'mobile' in jsx_code.lower()
        
        return {
            "complexity": complexity,
            "reusability": reusability,
            "mobile_optimized": mobile_optimized
        }
    
    def _generate_search_keywords(self, category: str, image_patterns: Dict, 
                                text_patterns: Dict, layout_patterns: Dict) -> str:
        """검색 키워드 생성"""
        keywords = [
            category,
            f"images_{image_patterns['count']}",
            image_patterns['arrangement'],
            layout_patterns['method'],
            text_patterns['alignment']
        ]
        
        keywords.extend(text_patterns['hierarchy'])
        
        return " ".join(keywords)
    
    def _create_jsx_embedding_text(self, analysis: Dict) -> str:
        """JSX 분석 결과를 임베딩 텍스트로 변환"""
        embedding_components = [
            f"Component category: {analysis['category']}",
            f"Layout method: {analysis['layout']['method']}",
            f"Images: {analysis['images']['count']} {analysis['images']['arrangement']}",
            f"Text hierarchy: {' '.join(analysis['text']['hierarchy'])}",
            f"Responsive: {analysis['layout']['responsive']}",
            f"Complexity: {analysis['metadata']['complexity']}",
            f"Alignment: {analysis['text']['alignment']}",
            f"Keywords: {analysis['keywords']}"
        ]
        
        return " ".join(embedding_components)
    
    def _create_jsx_embedding(self, text: str) -> List[float]:
        """JSX 임베딩 생성"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ JSX 임베딩 생성 실패: {e}")
            return [0.0] * 1536
    
    def _create_safe_jsx_key(self, component_name: str) -> str:
        """안전한 JSX 문서 키 생성"""
        safe_name = re.sub(r'[^a-zA-Z0-9_\-=]', '_', component_name)
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        return safe_name[:100] if len(safe_name) > 100 else safe_name
    
    def _upload_jsx_to_search_index(self, documents: List[Dict]):
        """JSX 문서를 검색 인덱스에 업로드"""
        try:
            result = self.search_client.upload_documents(documents)
            print(f"✅ {len(documents)}개 JSX 문서 업로드 완료")
            
            # 업로드 결과 확인
            failed_count = 0
            for res in result:
                if not res.succeeded:
                    failed_count += 1
                    print(f"❌ 업로드 실패: {res.key} - {res.error_message}")
            
            if failed_count == 0:
                print(f"🎉 모든 JSX 문서 업로드 성공!")
            else:
                print(f"⚠️ {failed_count}개 JSX 문서 업로드 실패")
                
        except Exception as e:
            print(f"❌ JSX 문서 업로드 실패: {e}")
    
    def _get_jsx_document_count(self) -> int:
        """JSX 인덱스의 총 문서 수 반환"""
        try:
            results = self.search_client.search(
                search_text="*",
                top=0,
                include_total_count=True
            )
            total_count = getattr(results, 'get_count', lambda: 0)()
            return total_count if total_count else 0
        except Exception as e:
            print(f"JSX 문서 수 확인 중 오류: {e}")
            return 0
    
    def search_jsx_components(self, query_text: str, category: str = None, 
                            image_count: int = None, complexity: str = None, 
                            top_k: int = 5) -> List[Dict]:
        """JSX 컴포넌트 검색"""
        try:
            # 쿼리 텍스트를 벡터로 변환
            query_embedding = self._create_jsx_embedding(query_text)
            
            # 벡터 검색 쿼리 생성
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k * 2,
                fields="jsx_vector"
            )
            
            # 검색 파라미터
            search_params = {
                "vector_queries": [vector_query],
                "top": top_k * 2,
                "select": [
                    "id", "component_name", "component_category", "jsx_structure",
                    "layout_method", "image_count", "image_arrangement", 
                    "complexity_level", "jsx_code", "search_keywords"
                ]
            }
            
            # 필터 조건 추가
            filters = []
            if category:
                filters.append(f"component_category eq '{category}'")
            if image_count is not None:
                filters.append(f"image_count eq {image_count}")
            if complexity:
                filters.append(f"complexity_level eq '{complexity}'")
            
            if filters:
                search_params["filter"] = " and ".join(filters)
            
            # 검색 실행
            results = self.search_client.search(**search_params)
            
            jsx_components = []
            for result in results:
                component_data = {
                    "id": result["id"],
                    "component_name": result["component_name"],
                    "category": result["component_category"],
                    "layout_method": result["layout_method"],
                    "image_count": result["image_count"],
                    "image_arrangement": result["image_arrangement"],
                    "complexity": result["complexity_level"],
                    "jsx_code": result["jsx_code"],
                    "keywords": result["search_keywords"],
                    "score": result.get("@search.score", 0.0)
                }
                jsx_components.append(component_data)
            
            return jsx_components[:top_k]
            
        except Exception as e:
            print(f"❌ JSX 컴포넌트 검색 실패: {e}")
            return []
    
    def get_jsx_recommendations(self, content_description: str, 
                              image_count: int = None, layout_preference: str = None) -> List[Dict]:
        """콘텐츠 설명을 바탕으로 JSX 컴포넌트 추천"""
        
        # 이미지 수에 따른 카테고리 추천
        if image_count is not None:
            if image_count == 0:
                category = "text_focused"
            elif image_count <= 2:
                category = "mixed"
            else:
                category = "image_focused"
        else:
            category = None
        
        # 검색 쿼리 구성
        search_query = f"{content_description} layout design component"
        if layout_preference:
            search_query += f" {layout_preference}"
        
        return self.search_jsx_components(
            query_text=search_query,
            category=category,
            image_count=image_count,
            top_k=3
        )

# 사용 예시 및 테스트 코드
if __name__ == "__main__":
    jsx_manager = JSXVectorManager()
    
    print("🚀 JSX 벡터 인덱스 생성 시작")
    
    # JSX 컴포넌트 처리
    jsx_manager.process_jsx_components("components")
    
    print("\n🔍 JSX 컴포넌트 검색 테스트")
    
    # 검색 테스트
    results = jsx_manager.search_jsx_components(
        query_text="magazine layout with images",
        category="image_focused",
        top_k=3
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['component_name']} (점수: {result['score']:.3f})")
        print(f"   카테고리: {result['category']}, 이미지: {result['image_count']}개")
    
    print("\n🎯 JSX 컴포넌트 추천 테스트")
    
    # 추천 테스트
    recommendations = jsx_manager.get_jsx_recommendations(
        content_description="travel magazine article with photos",
        image_count=2,
        layout_preference="grid"
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['component_name']} - {rec['layout_method']} 레이아웃")
