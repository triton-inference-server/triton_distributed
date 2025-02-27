package controllers

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"os"

	"github.com/gin-gonic/gin"
)

type proxyController struct{}

var ProxyController = proxyController{}

var NDS_HOST = os.Getenv("NDS_HOST")
var NDS_PORT = os.Getenv("NDS_PORT")

func (*proxyController) ReverseProxy(c *gin.Context) {
	director := func(req *http.Request) {
		r := c.Request

		req.URL.Scheme = "http"
		req.URL.Host = fmt.Sprintf("%s:%s", NDS_HOST, NDS_PORT)
		req.Header = r.Header.Clone()
	}
	proxy := &httputil.ReverseProxy{Director: director}
	proxy.ServeHTTP(c.Writer, c.Request)
}
