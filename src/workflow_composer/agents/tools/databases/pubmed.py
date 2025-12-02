"""
PubMed Database Client
======================

Client for the NCBI Entrez E-utilities API to search PubMed
and retrieve literature citations.

PubMed comprises more than 35 million citations for biomedical
literature from MEDLINE, life science journals, and online books.

API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25499/

Example:
    >>> from workflow_composer.agents.tools.databases import PubMedClient
    >>> 
    >>> client = PubMedClient()
    >>> result = client.search("CRISPR gene editing", limit=10)
    >>> print(result.count)
    10
    >>> print(result.data[0]["title"])
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

from .base import DatabaseClient, DatabaseResult

logger = logging.getLogger(__name__)


class PubMedClient(DatabaseClient):
    """
    Client for NCBI Entrez E-utilities (PubMed).
    
    Provides access to:
    - PubMed literature search
    - Article abstracts and metadata
    - MeSH term lookups
    - Citation data
    
    Note: For high-volume usage, register for an NCBI API key
    and set the api_key parameter or NCBI_API_KEY environment variable.
    
    Rate limits:
    - Without API key: 3 requests/second
    - With API key: 10 requests/second
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    NAME = "PubMed"
    RATE_LIMIT = 3.0  # Without API key
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize PubMed client.
        
        Args:
            api_key: NCBI API key for higher rate limits
            email: Contact email (recommended by NCBI)
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        
        self.api_key = api_key
        self.email = email
        
        # Increase rate limit if API key provided
        if api_key:
            self.RATE_LIMIT = 10.0
        
        # Try to load from environment
        if not self.api_key:
            import os
            self.api_key = os.environ.get("NCBI_API_KEY")
        
        if not self.email:
            import os
            self.email = os.environ.get("NCBI_EMAIL")
    
    def _add_api_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add API key and email to parameters."""
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params
    
    def search(
        self,
        query: str,
        limit: int = 20,
        sort: str = "relevance",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        **kwargs,
    ) -> DatabaseResult:
        """
        Search PubMed for articles.
        
        Args:
            query: Search terms (supports PubMed query syntax)
            limit: Maximum results to return
            sort: Sort order: "relevance", "pub_date", "first_author"
            date_from: Filter by date (YYYY/MM/DD or YYYY)
            date_to: Filter by date (YYYY/MM/DD or YYYY)
            
        Returns:
            DatabaseResult with article IDs
            
        Example:
            >>> result = client.search("cancer AND machine learning[Title]")
            >>> pmids = [r["pmid"] for r in result.data]
        """
        # Build query with date filters
        full_query = query
        if date_from or date_to:
            date_filter = f"{date_from or '1900/01/01'}:{date_to or '3000/01/01'}[dp]"
            full_query = f"({query}) AND {date_filter}"
        
        try:
            # Search for PMIDs
            params = self._add_api_params({
                "db": "pubmed",
                "term": full_query,
                "retmax": limit,
                "retmode": "json",
                "sort": sort,
                "usehistory": "y",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/esearch.fcgi",
                params=params,
            )
            
            result = response.get("esearchresult", {})
            pmids = result.get("idlist", [])
            total_count = int(result.get("count", 0))
            
            # Fetch article details
            if pmids:
                articles = self.fetch_articles(pmids)
            else:
                articles = []
            
            return DatabaseResult(
                success=True,
                data=articles,
                count=len(articles),
                query=query,
                source=self.NAME,
                message=f"Found {total_count} articles, returned {len(articles)}",
                metadata={
                    "total_count": total_count,
                    "webenv": result.get("webenv"),
                    "query_key": result.get("querykey"),
                },
            )
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return self._error_result(query, e)
    
    def get_by_id(
        self,
        pmid: str,
        **kwargs,
    ) -> DatabaseResult:
        """
        Get article by PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            DatabaseResult with article details
        """
        articles = self.fetch_articles([pmid])
        
        if articles:
            return DatabaseResult(
                success=True,
                data=articles[0],
                count=1,
                query=pmid,
                source=self.NAME,
                message=f"Retrieved article {pmid}",
            )
        else:
            return self._empty_result(pmid, f"Article {pmid} not found")
    
    def fetch_articles(
        self,
        pmids: List[str],
        include_abstract: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch article details for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            include_abstract: Include article abstracts
            
        Returns:
            List of article dictionaries
        """
        if not pmids:
            return []
        
        try:
            params = self._add_api_params({
                "db": "pubmed",
                "id": ",".join(str(p) for p in pmids),
                "retmode": "xml",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/efetch.fcgi",
                params=params,
            )
            
            # Parse XML response
            return self._parse_pubmed_xml(response, include_abstract)
            
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            return []
    
    def _parse_pubmed_xml(
        self,
        xml_text: str,
        include_abstract: bool = True,
    ) -> List[Dict[str, Any]]:
        """Parse PubMed XML response into article dictionaries."""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                article_data = {}
                
                # Get PMID
                pmid_elem = article.find(".//PMID")
                article_data["pmid"] = pmid_elem.text if pmid_elem is not None else ""
                
                # Get article info
                medline = article.find(".//MedlineCitation")
                if medline is None:
                    continue
                
                article_elem = medline.find(".//Article")
                if article_elem is None:
                    continue
                
                # Title
                title_elem = article_elem.find(".//ArticleTitle")
                article_data["title"] = title_elem.text if title_elem is not None else ""
                
                # Abstract
                if include_abstract:
                    abstract_elem = article_elem.find(".//Abstract/AbstractText")
                    if abstract_elem is not None:
                        article_data["abstract"] = abstract_elem.text or ""
                    else:
                        # Try to concatenate multiple abstract parts
                        abstract_parts = article_elem.findall(".//Abstract/AbstractText")
                        if abstract_parts:
                            abstract_texts = []
                            for part in abstract_parts:
                                label = part.get("Label", "")
                                text = part.text or ""
                                if label:
                                    abstract_texts.append(f"{label}: {text}")
                                else:
                                    abstract_texts.append(text)
                            article_data["abstract"] = " ".join(abstract_texts)
                        else:
                            article_data["abstract"] = ""
                
                # Authors
                authors = []
                for author in article_elem.findall(".//Author"):
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None:
                        name = last_name.text or ""
                        if fore_name is not None and fore_name.text:
                            name = f"{name} {fore_name.text}"
                        authors.append(name)
                article_data["authors"] = authors
                
                # Journal info
                journal = article_elem.find(".//Journal")
                if journal is not None:
                    journal_title = journal.find(".//Title")
                    article_data["journal"] = journal_title.text if journal_title is not None else ""
                    
                    # Publication date
                    pub_date = journal.find(".//PubDate")
                    if pub_date is not None:
                        year = pub_date.find("Year")
                        month = pub_date.find("Month")
                        day = pub_date.find("Day")
                        
                        year_str = year.text if year is not None else ""
                        month_str = month.text if month is not None else ""
                        day_str = day.text if day is not None else ""
                        
                        article_data["pub_date"] = f"{year_str} {month_str} {day_str}".strip()
                        article_data["year"] = year_str
                
                # MeSH terms
                mesh_terms = []
                for mesh in article.findall(".//MeshHeading/DescriptorName"):
                    if mesh.text:
                        mesh_terms.append(mesh.text)
                article_data["mesh_terms"] = mesh_terms
                
                # DOI
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        article_data["doi"] = article_id.text
                        break
                
                articles.append(article_data)
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
        
        return articles
    
    def search_related(
        self,
        pmid: str,
        limit: int = 20,
    ) -> DatabaseResult:
        """
        Find related articles for a given PMID.
        
        Args:
            pmid: PubMed ID
            limit: Maximum results
            
        Returns:
            DatabaseResult with related articles
        """
        try:
            params = self._add_api_params({
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "linkname": "pubmed_pubmed",
                "retmode": "json",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/elink.fcgi",
                params=params,
            )
            
            # Extract related PMIDs
            related_pmids = []
            linksets = response.get("linksets", [])
            for linkset in linksets:
                for linksetdb in linkset.get("linksetdbs", []):
                    if linksetdb.get("linkname") == "pubmed_pubmed":
                        related_pmids = [
                            str(link["id"]) 
                            for link in linksetdb.get("links", [])
                        ][:limit]
                        break
            
            # Fetch article details
            if related_pmids:
                articles = self.fetch_articles(related_pmids)
            else:
                articles = []
            
            return DatabaseResult(
                success=True,
                data=articles,
                count=len(articles),
                query=f"related to {pmid}",
                source=self.NAME,
                message=f"Found {len(articles)} related articles",
            )
            
        except Exception as e:
            logger.error(f"Related article search failed: {e}")
            return self._error_result(f"related to {pmid}", e)
    
    def get_citations(
        self,
        pmid: str,
        limit: int = 100,
    ) -> DatabaseResult:
        """
        Get articles that cite the given PMID.
        
        Args:
            pmid: PubMed ID
            limit: Maximum citations to return
            
        Returns:
            DatabaseResult with citing articles
        """
        try:
            params = self._add_api_params({
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "linkname": "pubmed_pubmed_citedin",
                "retmode": "json",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/elink.fcgi",
                params=params,
            )
            
            # Extract citing PMIDs
            citing_pmids = []
            linksets = response.get("linksets", [])
            for linkset in linksets:
                for linksetdb in linkset.get("linksetdbs", []):
                    if "citedin" in linksetdb.get("linkname", ""):
                        citing_pmids = [
                            str(link["id"]) 
                            for link in linksetdb.get("links", [])
                        ][:limit]
                        break
            
            # Fetch article details
            if citing_pmids:
                articles = self.fetch_articles(citing_pmids)
            else:
                articles = []
            
            return DatabaseResult(
                success=True,
                data=articles,
                count=len(articles),
                query=f"citations of {pmid}",
                source=self.NAME,
                message=f"Found {len(articles)} citing articles",
            )
            
        except Exception as e:
            logger.error(f"Citation search failed: {e}")
            return self._error_result(f"citations of {pmid}", e)
    
    def search_by_mesh(
        self,
        mesh_term: str,
        limit: int = 20,
    ) -> DatabaseResult:
        """
        Search by MeSH (Medical Subject Heading) term.
        
        Args:
            mesh_term: MeSH term
            limit: Maximum results
            
        Returns:
            DatabaseResult with matching articles
        """
        query = f"{mesh_term}[MeSH Terms]"
        return self.search(query, limit=limit)
    
    def search_by_author(
        self,
        author_name: str,
        limit: int = 20,
    ) -> DatabaseResult:
        """
        Search for articles by author.
        
        Args:
            author_name: Author name (e.g., "Smith J" or "Smith JA")
            limit: Maximum results
            
        Returns:
            DatabaseResult with articles by the author
        """
        query = f"{author_name}[Author]"
        return self.search(query, limit=limit, sort="pub_date")
    
    def get_article_summary(self, pmid: str) -> Optional[Dict[str, str]]:
        """
        Get a concise summary of an article.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Dictionary with title, authors, abstract, etc.
        """
        result = self.get_by_id(pmid)
        
        if not result.success or not result.data:
            return None
        
        article = result.data
        
        # Format authors
        authors_str = ", ".join(article.get("authors", [])[:3])
        if len(article.get("authors", [])) > 3:
            authors_str += ", et al."
        
        return {
            "pmid": article.get("pmid", ""),
            "title": article.get("title", ""),
            "authors": authors_str,
            "journal": article.get("journal", ""),
            "year": article.get("year", ""),
            "abstract": article.get("abstract", "")[:500] + "..." if len(article.get("abstract", "")) > 500 else article.get("abstract", ""),
            "doi": article.get("doi", ""),
            "pubmed_link": f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', '')}/",
        }
    
    def search_recent(
        self,
        query: str,
        days: int = 30,
        limit: int = 20,
    ) -> DatabaseResult:
        """
        Search for recent articles (within specified days).
        
        Args:
            query: Search terms
            days: Number of days to look back
            limit: Maximum results
            
        Returns:
            DatabaseResult with recent articles
        """
        # Use reldate parameter for relative date
        try:
            params = self._add_api_params({
                "db": "pubmed",
                "term": query,
                "reldate": days,
                "datetype": "edat",  # Entrez date
                "retmax": limit,
                "retmode": "json",
                "sort": "pub_date",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/esearch.fcgi",
                params=params,
            )
            
            result = response.get("esearchresult", {})
            pmids = result.get("idlist", [])
            
            # Fetch article details
            if pmids:
                articles = self.fetch_articles(pmids)
            else:
                articles = []
            
            return DatabaseResult(
                success=True,
                data=articles,
                count=len(articles),
                query=query,
                source=self.NAME,
                message=f"Found {len(articles)} articles from last {days} days",
            )
            
        except Exception as e:
            logger.error(f"Recent article search failed: {e}")
            return self._error_result(query, e)
