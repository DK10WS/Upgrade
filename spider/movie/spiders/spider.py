import re

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class crawler(CrawlSpider):
    name = "moviebot"
    allowed_domains = ["allmovie.com"]
    start_urls = ["https://www.allmovie.com/"]

    rules = (
        Rule(
            LinkExtractor(
                allow="movie/",
                deny=[
                    "/streams",
                    "/user-reviews",
                    "/awards",
                    "/streams",
                    "/related",
                    "/cast-crew",
                ],
            ),
            callback="parse_movie",
            follow=True,
        ),
    )

    def parse_movie(self, response):
        title = response.css("h2.movie-title::text").get(default="").strip()
        generes = response.css(".header-movie-genres a::text").getall()
        sub_genres = response.css(".header-movie-subgenres a::text").getall()
        summary = response.css(".text p::text").get(default="").strip()
        rating = response.css(".allmovie-rating::text").get(default="").strip()

        runtime = (
            response.css('span:contains("Release Date") + span span::text')
            .get(default="")
            .strip()
        )
        run_time_cleaned = runtime.replace(" min.", "") if runtime else ""

        release_date = response.css(
            "span + span span::text").get(default="").strip()

        tags = response.css("div.charactList::text").getall()
        new_tags = [
            element.strip().replace("|", "").strip()
            for element in tags
            if element.strip()
            and not re.search(r"\$\d+(?:,\d{3})*", element)
            and element.strip().replace("|", "").strip()
        ]
        if "," in new_tags:
            new_tags.remove(",")

        yield {
            "title": title,
            "genres": generes,
            "sub_genres": sub_genres,
            "summary": summary,
            "rating": rating,
            "release_date": release_date,
            "runtime_in_mins": run_time_cleaned,
            "tags": new_tags,
        }
